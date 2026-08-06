"""Chat-level narratives: the group's own history, member arcs, pairs, eras.

`analyze_chat.py` answers "how much" and "who most". This module answers
questions with a shape to them: what the group called itself in 2019, how one
member's vocabulary moved year to year, which pair drifted apart, and where one
period of the chat ends and the next begins.

Nothing here writes files or builds HTML. Every function takes normalized
messages and returns plain data, so the report, the JSON and the tests all see
the same numbers.
"""
import re
from collections import Counter, defaultdict

import analyze_chat as ac

# A member's message counts as answered if anyone else speaks inside this
# window, and as a reply to whoever spoke last. Same windows the rest of the
# report uses, so "pair dynamics" and "relationships" cannot disagree.
REPLY_WINDOW_SECONDS = ac.REPLY_WINDOW_SECONDS
SESSION_GAP_SECONDS = ac.CONVERSATION_WINDOW_SECONDS
SILENCE_SECONDS = 24 * 60 * 60

MIN_ERA_MONTHS = 6
ERA_VOLUME_RATIO = 2.0
ERA_TOPIC_SURVIVAL = 1 / 3.0
ERA_TOPIC_WORDS = 20
ERA_WORD_FLOOR = 5
# A quarter too small to have a character of its own cannot open an era. On a
# sparse chat every rule fires otherwise: three messages against eight is a
# volume collapse, and two quarters of six words share no vocabulary.
ERA_MIN_QUARTER = 30
TURNOVER_MIN_USES = 5
# Drift is only meaningful between people who were actually talking: without a
# floor, a pair with three interactions in 2019 and none since reads as a
# 100% collapse and fills the page.
DRIFT_MIN_INTERACTIONS = 100


# --------------------------------------------------------------------------- #
# the group's own history                                                      #
# --------------------------------------------------------------------------- #

def _event(pattern):
    return re.compile(pattern + r"\s*$", re.IGNORECASE)


# Messenger narrates admin events into `content`, attributed to whoever did
# them. analyze_chat drops them from the vocabulary with SYSTEM_EVENT_RE; the
# same lines read back through these capturing patterns are a record of the
# group renaming and re-nicknaming itself for years. Order matters: "cleared"
# is checked before "set", and the group patterns before the nickname ones.
_EVENT_PATTERNS = [
    ("group_name", _event(r"^(?P<actor>.{1,80}?) named the group (?P<value>.+?)\.?")),
    ("group_name", _event(r"^(?P<actor>.{1,80}?) changed the group name to (?P<value>.+?)\.?")),
    ("nickname_clear", _event(r"^(?P<actor>.{1,80}?) cleared the nickname for (?P<target>.+?)\.?")),
    ("nickname_clear", _event(r"^(?P<actor>.{1,80}?) cleared (?:your|their own) nickname\.?")),
    ("nickname", _event(r"^(?P<actor>.{1,80}?) set the nickname for (?P<target>.+?) to (?P<value>.+?)\.?")),
    ("nickname", _event(r"^(?P<actor>.{1,80}?) set (?:their own|his own|her own) nickname to (?P<value>.+?)\.?")),
    ("nickname_you", _event(r"^(?P<actor>.{1,80}?) set your nickname to (?P<value>.+?)\.?")),
    ("photo", _event(r"^(?P<actor>.{1,80}?) changed the group photo\.?")),
    ("theme", _event(r"^(?P<actor>.{1,80}?) changed the theme(?: to (?P<value>.+?))?\.?")),
    ("quick_reaction", _event(r"^(?P<actor>.{1,80}?) set the quick reaction to (?P<value>.+?)\.?")),
    ("added", _event(r"^(?P<actor>.{1,80}?) added (?P<target>.{1,80}?) to the group\.?")),
    ("removed", _event(r"^(?P<actor>.{1,80}?) removed (?P<target>.{1,80}?) from the group\.?")),
    ("left", _event(r"^(?P<actor>.{1,80}?) left the group\.?")),
    ("joined", _event(r"^(?P<actor>.{1,80}?) joined the group\.?")),
]

_EVENT_LABELS = {
    "group_name": "renamed the group", "nickname": "set a nickname",
    "nickname_you": "set a nickname", "nickname_clear": "cleared a nickname",
    "photo": "changed the group photo", "theme": "changed the theme",
    "quick_reaction": "changed the quick reaction", "added": "added someone",
    "removed": "removed someone", "left": "left", "joined": "joined",
}


def parse_event(content):
    """One system message as (kind, fields), or None if it is ordinary text."""
    text = (content or "").strip()
    if not text or not ac.SYSTEM_EVENT_RE.match(text):
        return None
    for kind, pattern in _EVENT_PATTERNS:
        match = pattern.match(text)
        if match:
            fields = {k: v.strip() for k, v in match.groupdict().items() if v}
            if kind == "nickname" and "target" not in fields:
                fields["target"] = fields.get("actor", "")
            return kind, fields
    return None


def group_history(msgs):
    """Every rename and nickname the group gave itself, in order.

    A nickname is stored as a range rather than an event, because the question
    people actually ask is "what were we calling him in 2021", and that needs
    the span the name was current for, not the moment it was set.
    """
    names = []
    nicknames = defaultdict(list)
    membership = []
    actors = Counter()
    kinds = Counter()
    open_nick = {}
    for m in msgs:
        parsed = parse_event(m["content"])
        if parsed is None:
            continue
        kind, fields = parsed
        actor = fields.get("actor", "")
        actors[actor] += 1
        kinds[kind] += 1
        stamp = m["dt"].strftime("%Y-%m-%d")
        if kind == "group_name":
            if names:
                names[-1]["until"] = stamp
            names.append({"name": fields.get("value", ""), "by": actor,
                          "date": stamp, "ts": m["ts_ms"], "until": None})
        elif kind in ("nickname", "nickname_you"):
            # "set your nickname" names the exporting account, which the export
            # never says out loud, so it is counted but not put on a timeline.
            target = fields.get("target")
            if not target:
                continue
            if target in open_nick:
                open_nick[target]["until"] = stamp
            entry = {"nickname": fields.get("value", ""), "by": actor,
                     "from": stamp, "ts": m["ts_ms"], "until": None}
            nicknames[target].append(entry)
            open_nick[target] = entry
        elif kind == "nickname_clear":
            target = fields.get("target")
            if target and target in open_nick:
                open_nick.pop(target)["until"] = stamp
        elif kind in ("added", "removed", "left", "joined"):
            membership.append({"kind": kind, "actor": actor,
                               "target": fields.get("target", actor), "date": stamp})
    return {
        "names": names,
        "current_name": names[-1]["name"] if names else None,
        "nicknames": {k: v for k, v in sorted(nicknames.items())},
        "membership": membership,
        "busiest": actors,
        "kinds": kinds,
        "labels": _EVENT_LABELS,
        "total": sum(kinds.values()),
    }


# --------------------------------------------------------------------------- #
# who talks to whom                                                            #
# --------------------------------------------------------------------------- #

def talk_matrix(msgs):
    """Directed reply counts by year, from adjacency inside the reply window.

    An export with no message ids has no real reply edges, so "talked to" is
    the same approximation `pair_matrices` uses: you answered whoever spoke
    immediately before you, if they spoke recently enough.
    """
    per_year = defaultdict(lambda: defaultdict(Counter))
    for prev, cur in zip(msgs, msgs[1:]):
        if cur["sender"] == prev["sender"]:
            continue
        gap = (cur["ts_ms"] - prev["ts_ms"]) / 1000
        if 0 < gap <= REPLY_WINDOW_SECONDS:
            per_year[cur["dt"].year][cur["sender"]][prev["sender"]] += 1
    return per_year


def _next_other_speaker(msgs):
    """For each message, the index of the next message from someone else."""
    nxt = [None] * len(msgs)
    for i in range(len(msgs) - 2, -1, -1):
        nxt[i] = i + 1 if msgs[i + 1]["sender"] != msgs[i]["sender"] else nxt[i + 1]
    return nxt


def _full_years(msgs):
    """Years the export covers end to end, so a partial year is not a decline."""
    first, last = msgs[0]["dt"], msgs[-1]["dt"]
    years = []
    for year in range(first.year, last.year + 1):
        starts_before = (first.year < year) or (first.month == 1 and first.day == 1)
        ends_after = (last.year > year) or (last.month == 12 and last.day == 31)
        if starts_before and ends_after:
            years.append(year)
    return years


def relationships(msgs, top=10):
    """Pairs by year, drift between them, and who leaves whom on read."""
    if not msgs:
        return None
    members = sorted({m["sender"] for m in msgs})
    per_year = talk_matrix(msgs)
    pair_year = defaultdict(Counter)
    member_year = defaultdict(Counter)
    for year, senders in per_year.items():
        for sender, targets in senders.items():
            for target, n in targets.items():
                pair_year[tuple(sorted((sender, target)))][year] += n
                member_year[sender][year] += n
                member_year[target][year] += n
    for m in msgs:
        for actor, _ in m["reactions"]:
            if actor == m["sender"]:
                continue
            year = m["dt"].year
            pair_year[tuple(sorted((actor, m["sender"])))][year] += 1
            member_year[actor][year] += 1
            member_year[m["sender"]][year] += 1

    pairs = []
    for pair, years in pair_year.items():
        pairs.append({"pair": list(pair), "total": sum(years.values()),
                      "by_year": dict(years),
                      "peak_year": years.most_common(1)[0][0]})
    pairs.sort(key=lambda p: (-p["total"], p["pair"]))

    full = _full_years(msgs)
    recent = full[-1] if full else None
    drift = []
    if recent is not None:
        for p in pairs:
            peak = p["peak_year"]
            peak_n = p["by_year"].get(peak, 0)
            if peak == recent or peak_n < DRIFT_MIN_INTERACTIONS:
                continue
            moved = []
            for member in p["pair"]:
                peak_total = member_year[member].get(peak, 0)
                recent_total = member_year[member].get(recent, 0)
                # A share needs a denominator worth dividing by. Someone who
                # left and traded two messages in the recent year spent half
                # of their interaction on whoever they answered, which reads
                # as a friendship deepening.
                if min(peak_total, recent_total) < DRIFT_MIN_INTERACTIONS:
                    continue
                was = _share(peak_n, peak_total)
                now = _share(p["by_year"].get(recent, 0), recent_total)
                if was is None or now is None:
                    continue
                # Share of that member's own interaction, so a pair that only
                # looks quieter because the whole chat went quiet is not drift.
                change = (now - was) / was
                if abs(change) > 0.5:
                    moved.append({"pair": p["pair"], "member": member,
                                  "peak_year": peak, "recent_year": recent,
                                  "peak_interactions": peak_n,
                                  "was_pct": round(100 * was, 1),
                                  "now_pct": round(100 * now, 1),
                                  "change_pct": round(100 * change, 1)})
            # One row per pair. Both members move by definition when a pair
            # goes quiet, and listing the same drift twice reads as two.
            if moved:
                drift.append(max(moved, key=lambda d: abs(d["change_pct"])))
    # Biggest moves first, and among equal moves the pairs that had the most
    # going on, so a friendship ending outranks an acquaintance fading.
    drift.sort(key=lambda d: (-abs(d["change_pct"]), -d["peak_interactions"], d["pair"]))

    first_after_silence = Counter()
    last_word = Counter()
    for prev, cur in zip(msgs, msgs[1:]):
        gap = (cur["ts_ms"] - prev["ts_ms"]) / 1000
        if gap >= SILENCE_SECONDS:
            first_after_silence[cur["sender"]] += 1
        if gap > SESSION_GAP_SECONDS:
            last_word[prev["sender"]] += 1
    last_word[msgs[-1]["sender"]] += 1

    nxt = _next_other_speaker(msgs)
    unanswered = Counter()
    totals = Counter()
    for i, m in enumerate(msgs):
        totals[m["sender"]] += 1
        j = nxt[i]
        if j is None or (msgs[j]["ts_ms"] - m["ts_ms"]) / 1000 > REPLY_WINDOW_SECONDS:
            unanswered[m["sender"]] += 1
    ignored = [{"member": member, "unanswered": unanswered[member],
                "messages": totals[member],
                "pct": round(100 * unanswered[member] / totals[member], 1)}
               for member in members if totals[member]]
    # Sorted by the rate, not the count: the loudest member is left unanswered
    # most often simply by speaking most.
    ignored.sort(key=lambda r: (-r["pct"], r["member"]))

    return {"members": members, "pairs": pairs[:top], "all_pairs": pairs,
            "drift": drift, "full_years": full,
            "min_interactions": DRIFT_MIN_INTERACTIONS,
            "first_after_silence": first_after_silence, "last_word": last_word,
            "ignored": ignored[:top]}


def _share(part, whole):
    return (part / whole) if whole else None


# --------------------------------------------------------------------------- #
# one member's arc                                                             #
# --------------------------------------------------------------------------- #

def member_profile(msgs, member, talks=None, names=(), top=6):
    """One member's years: how much, about what, with whom, best received."""
    mine = [m for m in msgs if m["sender"] == member]
    if not mine:
        return None
    by_year = Counter(m["dt"].year for m in mine)
    talks = talks if talks is not None else talk_matrix(msgs)
    talks_to = {}
    closest = Counter()
    for year, senders in talks.items():
        row = senders.get(member)
        if not row:
            continue
        talks_to[year] = dict(row)
        closest.update(row)
    reacted = sorted(mine, key=lambda m: (-len(m["reactions"]), m["ts_ms"]))
    return {
        "member": member,
        "total": len(mine),
        "share": ac._pct(len(mine), len(msgs)),
        "first": mine[0]["dt"].strftime("%Y-%m-%d"),
        "last": mine[-1]["dt"].strftime("%Y-%m-%d"),
        "active_days": len({m["dt"].date() for m in mine}),
        "by_year": dict(by_year),
        "peak_year": by_year.most_common(1)[0][0],
        # tf-idf over this member's own years, so the words are what set their
        # 2021 apart from their 2022 rather than what the whole chat said.
        "words_by_year": ac.topic_words(mine, top=top, names=names)["by_year"],
        "talks_to": talks_to,
        "closest": closest.most_common(top),
        "top_reacted": [m for m in reacted[:5] if m["reactions"]],
    }


def member_profiles(msgs, names=(), top=6):
    """Every member's arc, sharing the one pass that builds the talk matrix."""
    talks = talk_matrix(msgs)
    counts = Counter(m["sender"] for m in msgs)
    return {member: member_profile(msgs, member, talks=talks, names=names, top=top)
            for member, _ in counts.most_common()}


# --------------------------------------------------------------------------- #
# eras and vocabulary turnover                                                 #
# --------------------------------------------------------------------------- #

def _months_between(first, last):
    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append("%04d-%02d" % (year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _month_words(msgs):
    per_month = defaultdict(Counter)
    for m in msgs:
        key = m["dt"].strftime("%Y-%m")
        for w in ac._vocab(m):
            if w in ac.STOPWORDS or len(w) <= 2:
                continue
            per_month[key][w] += 1
    return per_month


def eras(msgs, names=()):
    """The chat cut into periods, split where a quarter breaks from the last.

    The rule, in one sentence: a month opens a new era when the three months
    from it carry less than half or more than double the messages of the three
    before it, or when fewer than a third of the previous quarter's top words
    are still in the top words of this one.

    A rule anybody can restate is worth more here than a better fit nobody can
    argue with, so there is no clustering and no tuning beyond those two
    thresholds.
    """
    if not msgs:
        return None
    months = _months_between(msgs[0]["dt"], msgs[-1]["dt"])
    if len(months) < 2 * MIN_ERA_MONTHS:
        return {"eras": [], "months": months, "reason": "chat is too short to segment"}
    volume = Counter(m["dt"].strftime("%Y-%m") for m in msgs)
    words = _month_words(msgs)

    def quarter_words(chunk):
        counter = Counter()
        for key in chunk:
            counter.update(words.get(key, {}))
        return {w for w, _ in counter.most_common(ERA_TOPIC_WORDS)}

    # Scaled to the chat, so the same rule reads a lull in a million-message
    # chat and in a thousand-message one.
    floor = max(ERA_MIN_QUARTER, len(msgs) // 100)
    boundaries = []
    for i in range(3, len(months) - 2):
        before = months[i - 3:i]
        after = months[i:i + 3]
        vol_before = sum(volume[k] for k in before)
        vol_after = sum(volume[k] for k in after)
        if max(vol_before, vol_after) < floor:
            continue
        split = False
        if vol_before and vol_after:
            ratio = vol_after / vol_before
            split = ratio >= ERA_VOLUME_RATIO or ratio <= 1 / ERA_VOLUME_RATIO
        else:
            split = True
        if not split:
            old, new = quarter_words(before), quarter_words(after)
            # Both quarters need a full set of top words before the overlap
            # between them means anything.
            if len(old) >= ERA_TOPIC_WORDS and len(new) >= ERA_TOPIC_WORDS:
                split = len(old & new) / len(old) < ERA_TOPIC_SURVIVAL
        if split:
            boundaries.append(i)

    # Boundaries closer together than a spell of the chat can be are merged:
    # two splits a month apart describe one change, not two eras.
    starts = [0]
    for i in boundaries:
        if i - starts[-1] >= MIN_ERA_MONTHS and len(months) - i >= MIN_ERA_MONTHS:
            starts.append(i)
    spans = []
    for n, start in enumerate(starts):
        end = starts[n + 1] - 1 if n + 1 < len(starts) else len(months) - 1
        spans.append((months[start], months[end]))

    return {"eras": _describe_eras(msgs, spans, names), "months": months,
            "boundaries": [months[i] for i in boundaries], "reason": None}


def _describe_eras(msgs, spans, names=()):
    """Name each era after the word it uses most out of proportion.

    tf-idf was the obvious choice and it does not work here: with five eras a
    word used constantly in all of them keeps an idf of 1.0, so its enormous
    term frequency wins and four eras out of five come back named after the
    chat's single commonest word. Scoring each word against the share it holds
    in the whole chat instead makes an evenly-spread word land at 1.0 and drop
    out, which is what "distinctive" was supposed to mean.
    """
    index = dict(enumerate(spans))
    per_era = defaultdict(Counter)
    overall = Counter()
    totals = Counter()
    senders = defaultdict(Counter)
    name_words = ac._member_name_words(msgs, names)
    for m in msgs:
        key = m["dt"].strftime("%Y-%m")
        era = None
        for n, (start, end) in index.items():
            if start <= key <= end:
                era = n
                break
        if era is None:
            continue
        totals[era] += 1
        senders[era][m["sender"]] += 1
        for w in ac._vocab(m):
            if w in ac.STOPWORDS or w in name_words or len(w) <= 2:
                continue
            per_era[era][w] += 1
            overall[w] += 1
    all_words = max(1, sum(overall.values()))
    out = []
    for n, (start, end) in enumerate(spans):
        counter = per_era[n]
        era_words = max(1, sum(counter.values()))
        # A floor of its own size, so a word said twice in a short era cannot
        # out-score one the era is actually built on.
        floor = max(ERA_WORD_FLOOR, era_words // 20000)
        scored = []
        for w, c in counter.items():
            if c < floor:
                continue
            lift = (c / era_words) / (overall[w] / all_words)
            # A word spread evenly across the chat sits at 1.0 and says nothing
            # about this era. With only one era everything sits at 1.0, so
            # there is nothing to be distinctive against and the plain counts
            # are the honest answer.
            if lift <= 1.0 and len(spans) > 1:
                continue
            scored.append((round(lift, 2), c, w))
        # Words an era holds alone all reach the same lift, so the count breaks
        # the tie: among equally exclusive words the era is named for the one
        # it actually said most.
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        top_words = [{"word": w, "count": c, "lift": lift}
                     for lift, c, w in scored[:8]]
        out.append({
            "start": start, "end": end,
            "months": _month_count(start, end),
            "messages": totals[n],
            "name": top_words[0]["word"] if top_words else "quiet",
            "words": top_words,
            "top_member": senders[n].most_common(1)[0][0] if senders[n] else None,
        })
    return out


def _month_count(start, end):
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    return (ey - sy) * 12 + (em - sm) + 1


def vocabulary_turnover(msgs, min_uses=TURNOVER_MIN_USES, top=15, names=()):
    """Words the chat picked up in a year, and words it stopped saying.

    A word counts as born in the year it is first said and dead in the year it
    is last said, which makes the final year's "died" list meaningless — every
    word alive at the end dies there — so it is left out.
    """
    first_year = {}
    last_year = {}
    totals = Counter()
    name_words = ac._member_name_words(msgs, names)
    for m in msgs:
        year = m["dt"].year
        for w in ac._vocab(m):
            if w in ac.STOPWORDS or w in name_words or len(w) <= 2:
                continue
            totals[w] += 1
            if w not in first_year:
                first_year[w] = year
            last_year[w] = year
    years = sorted({m["dt"].year for m in msgs})
    born = defaultdict(list)
    died = defaultdict(list)
    for w, count in totals.items():
        if count < min_uses:
            continue
        if first_year[w] != years[0]:
            born[first_year[w]].append((count, w))
        if last_year[w] != years[-1]:
            died[last_year[w]].append((count, w))
    return {
        "years": years,
        "born": {y: [{"word": w, "count": c} for c, w in sorted(v, reverse=True)[:top]]
                 for y, v in born.items()},
        "died": {y: [{"word": w, "count": c} for c, w in sorted(v, reverse=True)[:top]]
                 for y, v in died.items()},
    }
