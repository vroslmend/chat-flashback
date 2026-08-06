"""An inverted index over a thread, and the word profiles built on top of it.

The index maps a word to the messages containing it and nothing else. Every
statistic is computed on demand by walking that matched subset, which for a
common word is tens of thousands of messages rather than the whole chat. The
cost of that choice is a walk per query; the benefit is that adding a new
statistic never means rebuilding or reshaping the index.

A phrase needs no index of its own: the messages that could contain it are the
ones containing every one of its words, which is an intersection of postings,
and adjacency is then verified message by message.
"""
import random
import re
from array import array
from collections import Counter, defaultdict

import analyze_chat as ac

_RUN_RE = re.compile(r"(.)\1+")
MAX_COLLOCATIONS = 10
MIN_COLLOCATION_MESSAGES = 3
RANDOM_EXAMPLES = 3


def _collapse(word):
    """"bruhhh" and "bruh" both collapse to "bruh"; "brhu" does not."""
    return _RUN_RE.sub(r"\1", word)


def _sequence(text):
    """Words and emoji in the order they were typed.

    tokenize() and split_emojis() each return their own stream, which is all a
    single token needs but loses the ordering a phrase has to be checked
    against. Both are re-derived here from the same regex and the same emoji
    library, over one lowercased string so their offsets agree, and a token is
    the same thing to a sequence as it is to the index.
    """
    lowered = (text or "").lower()
    items = [(m.start(), m.group()) for m in ac._WORD_RE.finditer(lowered)
             if any(c.isalpha() for c in m.group())]
    items += [(e["match_start"], e["emoji"])
              for e in ac.emoji_lib.emoji_list(lowered)]
    items.sort()
    return [token for _, token in items]


def _count_run(seq, pattern):
    """Occurrences of `pattern` as consecutive tokens of `seq`, overlaps included."""
    n = len(pattern)
    if n == 1:
        return seq.count(pattern[0])
    if n > len(seq):
        return 0
    first = pattern[0]
    hits = 0
    for i in range(len(seq) - n + 1):
        if seq[i] == first and tuple(seq[i:i + n]) == pattern:
            hits += 1
    return hits


class WordIndex:
    """word -> message indices, plus whole-chat totals for baselines."""

    def __init__(self, msgs, progress=None):
        self.msgs = msgs
        self.total_messages = len(msgs)
        self.totals = Counter()
        self.member_totals = Counter(m["sender"] for m in msgs)
        self.mean_reactions = (sum(len(m["reactions"]) for m in msgs) / len(msgs)
                               if msgs else 0.0)
        building = defaultdict(list)
        for i, m in enumerate(msgs):
            if progress is not None and i and i % 100000 == 0:
                progress(i)
            body = ac._words_only(m["content"])
            if not body:
                continue
            seen = set()
            for token in ac.tokenize(body) + ac.split_emojis(body):
                self.totals[token] += 1
                if token not in seen:
                    seen.add(token)
                    building[token].append(i)
        # array("i") holds a posting in 4 bytes rather than the ~28 a Python int
        # costs, which on a chat this size is the difference between tens and
        # hundreds of megabytes.
        self.postings = {w: array("i", idxs) for w, idxs in building.items()}
        # Grouped once at build time: scanning the whole vocabulary per query
        # would make every profile cost O(distinct words).
        self._by_collapsed = defaultdict(list)
        for w in self.postings:
            self._by_collapsed[_collapse(w)].append(w)

    # ----------------------------------------------------------------- #
    # queries                                                            #
    # ----------------------------------------------------------------- #

    def profile(self, query, fold_variants=False):
        """A word or a phrase, told apart by how many tokens the query holds."""
        terms = _sequence(query)
        if not terms:
            return None
        if len(terms) > 1:
            return self._phrase_profile(terms)
        word = terms[0]
        if word not in self.postings:
            return None
        words = [word]
        if fold_variants:
            words += [v["word"] for v in self.variants_of(word)]
        if len(words) == 1:
            idxs = list(self.postings[word])
        else:
            idxs = sorted({i for w in words for i in self.postings[w]})
        result = self._profile_from(word, idxs, [(w,) for w in words])
        result["variants"] = self.variants_of(word)
        result["folded"] = bool(fold_variants)
        return result

    def _phrase_profile(self, terms):
        """Messages holding every word, then those holding them side by side."""
        distinct = set(terms)
        if any(term not in self.postings for term in distinct):
            return None
        # Narrowed from the rarest word outward, so the verification pass that
        # follows reads a few hundred messages rather than every message the
        # commonest word in the phrase appears in.
        rarest_first = sorted(distinct, key=lambda t: len(self.postings[t]))
        candidates = set(self.postings[rarest_first[0]])
        for term in rarest_first[1:]:
            candidates &= set(self.postings[term])
            if not candidates:
                return None
        pattern = tuple(terms)
        # Verification already reads each candidate; the sequences of the ones
        # that survive are what the profile walk would otherwise rebuild.
        seqs = {}
        for i in sorted(candidates):
            seq = _sequence(ac._words_only(self.msgs[i]["content"]))
            if _count_run(seq, pattern):
                seqs[i] = seq
        if not seqs:
            return None
        result = self._profile_from(" ".join(terms), sorted(seqs), [pattern], seqs)
        result["variants"] = []
        result["folded"] = False
        return result

    def variants_of(self, word):
        """Other spellings that differ from `word` only in held-down letters."""
        family = self._by_collapsed.get(_collapse(word), [])
        out = [{"word": w, "uses": self.totals[w]} for w in family if w != word]
        out.sort(key=lambda v: (-v["uses"], v["word"]))
        return out

    def suggest(self, prefix, limit=10):
        """Completions of the last word, keeping any words typed before it.

        Only the last word is completed: whether the finished phrase was ever
        said is a question the profile answers, and answering it here would
        mean resolving a phrase on every keystroke.
        """
        # Only the left side is stripped: a trailing space means the next word
        # has not been typed yet, and there is nothing to complete.
        text = (prefix or "").lower().lstrip()
        head, _, tail = text.rpartition(" ")
        if not tail:
            return []
        hits = [w for w in self.postings if w.startswith(tail)]
        hits.sort(key=lambda w: (-self.totals[w], w))
        hits = hits[:limit]
        return [head + " " + w for w in hits] if head else hits

    # ----------------------------------------------------------------- #
    # the profile                                                        #
    # ----------------------------------------------------------------- #

    def _profile_from(self, label, idxs, patterns, seqs=None):
        """Every statistic, in one pass over the messages that matched.

        `patterns` is what counts as a use: one token sequence for a phrase,
        one per spelling when variants are folded together. `seqs` lets a
        caller that has already read those messages hand over what it read.
        """
        matched = [self.msgs[i] for i in idxs]
        wordset = {token for pattern in patterns for token in pattern}
        per_member_uses = Counter()
        per_member_msgs = Counter()
        by_year = Counter()
        by_month = Counter()
        by_hour = [0] * 24
        neighbours = Counter()
        alone = 0
        first_use = {}
        for pos, m in enumerate(matched):
            seq = (seqs[idxs[pos]] if seqs is not None
                   else _sequence(ac._words_only(m["content"])))
            uses = sum(_count_run(seq, pattern) for pattern in patterns)
            sender = m["sender"]
            per_member_uses[sender] += uses
            per_member_msgs[sender] += 1
            dt = m["dt"]
            by_year[dt.year] += uses
            by_month[dt.strftime("%Y-%m")] += uses
            by_hour[dt.hour] += uses
            present = set(seq)
            if present and present <= wordset:
                alone += 1
            for other in present - wordset:
                neighbours[other] += 1
            if sender not in first_use:
                first_use[sender] = (m, idxs[pos])

        per_member = []
        for member, uses in per_member_uses.most_common():
            total = self.member_totals[member]
            per_member.append({
                "member": member,
                "uses": uses,
                "messages": per_member_msgs[member],
                "per_1k": round(1000 * uses / total, 1) if total else 0.0,
            })

        return {
            "word": label,
            "terms": list(patterns[0]) if len(patterns) == 1 else [label],
            "is_phrase": len(patterns[0]) > 1,
            "uses": sum(per_member_uses.values()),
            "messages": len(idxs),
            "per_member": per_member,
            "by_year": dict(by_year),
            "by_month": dict(by_month),
            "by_hour": by_hour,
            "peak_year": by_year.most_common(1)[0][0] if by_year else None,
            "first": self._cite(matched[0], idxs[0]),
            "last": self._cite(matched[-1], idxs[-1]),
            "adoption": self._adoption(first_use, matched[0]["dt"]),
            "reaction_pull": self._reaction_pull(matched),
            "collocations": self._collocations(neighbours, len(matched)),
            "alone_pct": round(100 * alone / len(matched), 1),
            "examples": self._examples(label, matched, idxs),
        }

    def _adoption(self, first_use, patient_zero):
        """Who said it first, and how long everyone else took to pick it up."""
        out = []
        for member, pair in sorted(first_use.items(), key=lambda kv: kv[1][0]["ts_ms"]):
            m, idx = pair
            out.append({"member": member, "first": self._cite(m, idx),
                        "days_after": (m["dt"] - patient_zero).days})
        for member in sorted(set(self.member_totals) - set(first_use)):
            out.append({"member": member, "first": None, "days_after": None})
        return out

    def _reaction_pull(self, matched):
        """Reactions these messages draw against the chat's own average."""
        if not self.mean_reactions:
            return None
        mine = sum(len(m["reactions"]) for m in matched) / len(matched)
        return round(mine / self.mean_reactions, 2)

    def _collocations(self, neighbours, matched_count):
        """Words that show up beside this one more than chance would predict.

        A raw co-occurrence count just relists the chat's most common words, so
        each is scored against the share the whole chat would lead you to expect.
        """
        out = []
        for other, n in neighbours.items():
            if n < MIN_COLLOCATION_MESSAGES or other in ac.STOPWORDS:
                continue
            expected = len(self.postings[other]) / self.total_messages
            if not expected:
                continue
            ratio = (n / matched_count) / expected
            if ratio > 1:
                out.append({"word": other, "ratio": round(ratio, 1), "messages": n})
        out.sort(key=lambda c: (-c["ratio"], -c["messages"], c["word"]))
        return out[:MAX_COLLOCATIONS]

    def _examples(self, label, matched, idxs):
        best = max(range(len(matched)), key=lambda p: len(matched[p]["reactions"]))
        most_reacted = None
        taken = {0}
        if matched[best]["reactions"]:
            most_reacted = self._cite(matched[best], idxs[best])
            taken.add(best)
        # Drawn from what the fixed picks left over: a phrase said twice would
        # otherwise show its two messages four times.
        rest = [p for p in range(len(matched)) if p not in taken]
        # Seeded on the query, so reloading a profile does not reshuffle it.
        picker = random.Random(label)
        sample = picker.sample(rest, min(RANDOM_EXAMPLES, len(rest)))
        return {
            "first": self._cite(matched[0], idxs[0]),
            "most_reacted": most_reacted,
            "random": [self._cite(matched[p], idxs[p]) for p in sorted(sample)],
        }

    def _cite(self, m, idx=None):
        return {"sender": m["sender"], "dt": m["dt"].strftime("%Y-%m-%d %H:%M"),
                "content": m["content"], "ts": m["ts_ms"], "index": idx}
