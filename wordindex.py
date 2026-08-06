"""An inverted index over a thread, and the word profiles built on top of it.

The index maps a word to the messages containing it and nothing else. Every
statistic is computed on demand by walking that matched subset, which for a
common word is tens of thousands of messages rather than the whole chat. The
cost of that choice is a walk per query; the benefit is that adding a new
statistic never means rebuilding or reshaping the index.
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
        word = (query or "").strip().lower()
        if word not in self.postings:
            return None
        words = [word]
        if fold_variants:
            words += [v["word"] for v in self.variants_of(word)]
        if len(words) == 1:
            idxs = list(self.postings[word])
        else:
            idxs = sorted({i for w in words for i in self.postings[w]})
        result = self._profile_from(word, idxs, words)
        result["variants"] = self.variants_of(word)
        result["folded"] = bool(fold_variants)
        return result

    def variants_of(self, word):
        """Other spellings that differ from `word` only in held-down letters."""
        family = self._by_collapsed.get(_collapse(word), [])
        out = [{"word": w, "uses": self.totals[w]} for w in family if w != word]
        out.sort(key=lambda v: (-v["uses"], v["word"]))
        return out

    def suggest(self, prefix, limit=10):
        p = (prefix or "").strip().lower()
        if not p:
            return []
        hits = [w for w in self.postings if w.startswith(p)]
        hits.sort(key=lambda w: (-self.totals[w], w))
        return hits[:limit]

    # ----------------------------------------------------------------- #
    # the profile                                                        #
    # ----------------------------------------------------------------- #

    def _profile_from(self, word, idxs, words):
        matched = [self.msgs[i] for i in idxs]
        wordset = set(words)
        per_member_uses = Counter()
        per_member_msgs = Counter()
        by_year = Counter()
        by_month = Counter()
        by_hour = [0] * 24
        neighbours = Counter()
        alone = 0
        first_use = {}
        for pos, m in enumerate(matched):
            body = ac._words_only(m["content"])
            tokens = ac.tokenize(body)
            emojis = ac.split_emojis(body)
            uses = sum(tokens.count(w) + emojis.count(w) for w in words)
            sender = m["sender"]
            per_member_uses[sender] += uses
            per_member_msgs[sender] += 1
            dt = m["dt"]
            by_year[dt.year] += uses
            by_month[dt.strftime("%Y-%m")] += uses
            by_hour[dt.hour] += uses
            present = set(tokens) | set(emojis)
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
            "word": word,
            "uses": sum(self.totals[w] for w in words),
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
            "examples": self._examples(word, matched, idxs),
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

    def _examples(self, word, matched, idxs):
        best = max(range(len(matched)), key=lambda p: len(matched[p]["reactions"]))
        # Seeded on the word, so reloading a profile does not reshuffle it.
        picker = random.Random(word)
        sample = picker.sample(range(len(matched)), min(RANDOM_EXAMPLES, len(matched)))
        return {
            "first": self._cite(matched[0], idxs[0]),
            "most_reacted": (self._cite(matched[best], idxs[best])
                             if matched[best]["reactions"] else None),
            "random": [self._cite(matched[p], idxs[p]) for p in sorted(sample)],
        }

    def _cite(self, m, idx=None):
        return {"sender": m["sender"], "dt": m["dt"].strftime("%Y-%m-%d %H:%M"),
                "content": m["content"], "ts": m["ts_ms"], "index": idx}
