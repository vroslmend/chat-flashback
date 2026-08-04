#!/usr/bin/env python3
"""chat-flashback: turn a Messenger export into a report about a group chat.

Parses a Facebook Messenger JSON export (or a WhatsApp/Telegram folder), computes
the analytics, and writes charts plus a summary.md. Everything runs locally.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import emoji as emoji_lib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from better_profanity import Profanity

DEFAULT_OUTPUT = "output"
REPLY_WINDOW_SECONDS = 60 * 60
MESSAGE_FILE_RE = re.compile(r"^message_\d+\.json$")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "at", "for", "with", "by", "from", "up", "about",
    "into", "over", "after", "i", "me", "my", "we", "our", "us", "you", "your", "yours",
    "he", "she", "it", "his", "her", "them", "they", "their", "this", "that", "these",
    "those", "so", "just", "not", "no", "yes", "yeah", "do", "does", "did", "have",
    "has", "had", "will", "would", "can", "could", "should", "shall", "may", "might",
    "must", "im", "ive", "id", "youre", "youve", "youll", "theyre", "theyve", "theres",
    "whats", "whos", "dont", "doesnt", "didnt", "cant", "wont", "wouldnt", "couldnt",
    "shouldnt", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "its", "its",
    "it's", "i'm", "that's", "there's", "let's", "we're", "you're", "they're", "don't",
    "what's", "who's", "didn't", "doesn't", "can't", "won't", "wouldn't", "couldn't",
    "shouldn't", "isn't", "aren't", "i've", "you've", "i'd", "you'd", "he's", "she's",
    "it's", "that'll", "whats", "whos", "theres", "get", "got", "gonna", "wanna", "lets", "ok", "okay", "lol", "lmao", "omg", "haha",
    "uh", "um", "er", "like", "really", "very", "much", "also", "then", "there", "here",
    "when", "where", "who", "what", "why", "how", "all", "any", "some", "more", "most",
    "other", "one", "two", "back", "know", "think", "see", "go", "say", "good", "new",
    "time", "day", "year", "thing", "going", "make", "even", "still", "way", "well",
}

CENSOR_WORDS = {str(w).lower() for w in Profanity().CENSOR_WORDSET}

PALETTE = ["#5b8ff9", "#5ad8a6", "#f6bd16", "#e8684a", "#6dc8ec", "#9270ca",
           "#ff9d4d", "#269a99", "#ff99c3", "#9fe6b8"]


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #

def numeric_key(path):
    m = re.search(r"\d+", path.name)
    return int(m.group()) if m else 0


def find_thread_dirs(root):
    root = Path(root)
    if not root.exists():
        print(f"[error] {root} does not exist. Point --input at a thread folder, "
              f"a `messages/` folder, or a folder containing thread folders.")
        return []
    dirs = []
    if any(MESSAGE_FILE_RE.match(f.name) for f in root.iterdir()):
        dirs.append(root)
    else:
        for d in sorted(root.rglob("*")):
            if d.is_dir() and any(MESSAGE_FILE_RE.match(f.name) for f in d.iterdir()):
                dirs.append(d)
    seen = set()
    uniq = []
    for d in dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def decode_messenger_text(text):
    """Fix Messenger's double-encoded unicode (\\u00f0\\u009f... -> emoji)."""
    if not text or "\\u00" not in text:
        return text
    if all(ord(c) < 256 for c in text):
        try:
            return text.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return text


def load_thread(thread_dir):
    files = sorted(thread_dir.glob("message_*.json"), key=numeric_key)
    if not files:
        return None
    messages, participants, title = [], [], thread_dir.name
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            raw = re.sub(r"[\x00-\x1f\x7f]", " ", f.read_text(encoding="utf-8", errors="ignore"))
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [warn] skipping unreadable {f.name}: {exc}")
                continue
        messages.extend(data.get("messages", []))
        if data.get("title"):
            title = data["title"]
        if data.get("participants"):
            participants = [p.get("name") for p in data["participants"] if p.get("name")]
    return title, participants, messages


def normalize_messages(raw):
    msgs = []
    for m in raw:
        sender = m.get("sender_name")
        if not sender:
            continue
        ts_ms = m.get("timestamp_ms")
        if ts_ms is None:
            ts_sec = m.get("timestamp")
            if ts_sec is None:
                continue
            ts_ms = int(float(ts_sec) * 1000)
        content = m.get("content")
        if content is not None:
            content = decode_messenger_text(str(content))
        reactions = []
        for r in m.get("reactions") or []:
            reaction = decode_messenger_text(str(r.get("reaction") or ""))
            actor = r.get("actor") or r.get("reactor") or r.get("sender_name")
            if actor and reaction:
                reactions.append((actor, reaction))
        share = m.get("share") if isinstance(m.get("share"), dict) else None
        msgs.append({
            "sender": sender,
            "ts_ms": ts_ms,
            "dt": datetime.fromtimestamp(ts_ms / 1000),
            "content": content,
            "mtype": m.get("type", "Generic"),
            "reactions": reactions,
            "has_photo": bool(m.get("photos")),
            "has_sticker": bool(m.get("sticker")),
            "link": (share or {}).get("link"),
            "call_duration": m.get("call_duration"),
            "reply_to": m.get("reply_to_message_id"),
            "is_unsent": bool(m.get("is_unsent")),
        })
    msgs.sort(key=lambda x: x["ts_ms"])
    return msgs


def anonymize_map(msgs):
    counts = Counter(m["sender"] for m in msgs)
    names = sorted(counts, key=lambda n: (-counts[n], msgs[0]["sender"] != n))
    return {n: f"Person {chr(ord('A') + i)}" for i, n in enumerate(names)}


def apply_anonymization(msgs, mapping):
    for m in msgs:
        m["sender"] = mapping.get(m["sender"], "Person ?")
        m["reactions"] = [(mapping.get(a, "Person ?"), r) for a, r in m["reactions"]]
    return msgs


# --------------------------------------------------------------------------- #
# Core stats                                                                  #
# --------------------------------------------------------------------------- #

def tokenize(text):
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def split_emojis(text):
    return [c for c in (text or "") if emoji_lib.is_emoji(c)]


def longest_streak(dates):
    if not dates:
        return 0
    dates = sorted(set(dates))
    best, run = 1, 1
    for a, b in zip(dates, dates[1:]):
        run = run + 1 if (b - a).days == 1 else 1
        best = max(best, run)
    return best


def core_stats(msgs):
    words = Counter()
    emojis = Counter()
    per_member_words = defaultdict(Counter)
    per_member_emojis = defaultdict(Counter)
    member_msgs = Counter()
    by_hour = Counter()
    by_weekday = Counter()
    by_month = Counter()
    by_year = Counter()
    by_day = Counter()
    for m in msgs:
        member_msgs[m["sender"]] += 1
        by_hour[m["dt"].hour] += 1
        by_weekday[m["dt"].weekday()] += 1
        by_month[m["dt"].month] += 1
        by_year[m["dt"].year] += 1
        by_day[m["dt"].date()] += 1
        for w in tokenize(m["content"]):
            if w not in STOPWORDS and len(w) > 1:
                words[w] += 1
                per_member_words[m["sender"]][w] += 1
        for e in split_emojis(m["content"]):
            emojis[e] += 1
            per_member_emojis[m["sender"]][e] += 1
    media = sum(1 for m in msgs if m["has_photo"] or m["has_sticker"])
    links = sum(1 for m in msgs if m["link"])
    calls = sum(1 for m in msgs if m["mtype"] == "Call")
    call_seconds = sum(m["call_duration"] or 0 for m in msgs if m["mtype"] == "Call")
    return {
        "total": len(msgs),
        "member_msgs": member_msgs,
        "words": words,
        "emojis": emojis,
        "per_member_words": per_member_words,
        "per_member_emojis": per_member_emojis,
        "by_hour": by_hour,
        "by_weekday": by_weekday,
        "by_month": by_month,
        "by_year": by_year,
        "by_day": by_day,
        "longest_streak": longest_streak(list(by_day)),
        "media": media,
        "links": links,
        "calls": calls,
        "call_seconds": call_seconds,
    }


# --------------------------------------------------------------------------- #
# Analyses                                                                    #
# --------------------------------------------------------------------------- #

def yearly_recaps(msgs):
    by_year = defaultdict(list)
    for m in msgs:
        by_year[m["dt"].year].append(m)
    recaps = {}
    for year, group in sorted(by_year.items()):
        member_msgs = Counter(m["sender"] for m in group)
        words = Counter()
        emojis = Counter()
        day_counts = Counter()
        best_reacted = None
        for m in group:
            words.update(w for w in tokenize(m["content"]) if w not in STOPWORDS and len(w) > 1)
            emojis.update(split_emojis(m["content"]))
            day_counts[m["dt"].date()] += 1
            if m["reactions"] and (best_reacted is None or
                                   len(m["reactions"]) > len(best_reacted["reactions"])):
                best_reacted = m
        record_day, record_count = day_counts.most_common(1)[0] if day_counts else (None, 0)
        recaps[year] = {
            "total": len(group),
            "active_members": len(member_msgs),
            "top_member": member_msgs.most_common(1)[0][0] if member_msgs else None,
            "top_word": words.most_common(1)[0] if words else None,
            "top_emoji": emojis.most_common(1)[0] if emojis else None,
            "record_day": record_day,
            "record_day_count": record_count,
            "best_reacted": best_reacted,
        }
    return recaps


def personalities(msgs, top=10):
    profiles = {}
    per_member_words = defaultdict(Counter)
    per_member_emojis = defaultdict(Counter)
    word_totals = Counter()
    per_member = defaultdict(list)
    for m in msgs:
        per_member[m["sender"]].append(m)
        for w in tokenize(m["content"]):
            if w not in STOPWORDS and len(w) > 1:
                per_member_words[m["sender"]][w] += 1
                word_totals[w] += 1
        for e in split_emojis(m["content"]):
            per_member_emojis[m["sender"]][e] += 1
    for member, group in per_member.items():
        total_msgs = len(group)
        total_words = sum(len(tokenize(m["content"])) for m in group)
        by_hour = Counter(m["dt"].hour for m in group)
        night = sum(1 for m in group if m["dt"].hour < 6)
        emojis = per_member_emojis[member]
        sig = []
        for w, c in per_member_words[member].items():
            if word_totals[w] >= 3:
                sig.append((c / word_totals[w], w, c))
        sig.sort(reverse=True)
        profiles[member] = {
            "total_msgs": total_msgs,
            "total_words": total_words,
            "avg_words": round(total_words / total_msgs, 2) if total_msgs else 0,
            "peak_hour": by_hour.most_common(1)[0][0] if by_hour else None,
            "night_pct": round(100 * night / total_msgs, 1) if total_msgs else 0,
            "top_words": [w for _, w, _ in sig[:top]],
            "signature": sig[0][1] if sig else None,
            "top_emojis": [e for e, _ in emojis.most_common(5)],
        }
    return profiles


def reaction_stats(msgs, top=10):
    reacted = [m for m in msgs if m["reactions"]]
    reactor = Counter()
    reaction_emoji = Counter()
    per_member_emoji = defaultdict(Counter)
    for m in reacted:
        for actor, reac in m["reactions"]:
            reactor[actor] += 1
            reaction_emoji[reac] += 1
            per_member_emoji[actor][reac] += 1
    top_messages = sorted(reacted, key=lambda m: -len(m["reactions"]))[:top]
    return {
        "total_reacted": len(reacted),
        "total_reactions": sum(len(m["reactions"]) for m in reacted),
        "reactor": reactor,
        "reaction_emoji": reaction_emoji,
        "per_member_emoji": per_member_emoji,
        "top_messages": top_messages,
        "most_reactive": reactor.most_common(1)[0][0] if reactor else None,
    }


def response_speed(msgs, top=10):
    reply_seconds = defaultdict(list)
    replies_received = Counter()
    messages_sent = Counter(m["sender"] for m in msgs)
    ignored = Counter()
    for prev, cur in zip(msgs, msgs[1:]):
        if cur["sender"] == prev["sender"]:
            continue
        gap = (cur["ts_ms"] - prev["ts_ms"]) / 1000
        if 0 < gap <= REPLY_WINDOW_SECONDS:
            reply_seconds[cur["sender"]].append(gap)
            replies_received[prev["sender"]] += 1
        else:
            ignored[prev["sender"]] += 1
    table = []
    for member, gaps in reply_seconds.items():
        gaps.sort()
        n = len(gaps)
        med = gaps[n // 2] if n % 2 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2
        quick = sum(1 for g in gaps if g <= 300)
        table.append({
            "member": member,
            "replies": n,
            "median_s": med,
            "median_m": round(med / 60, 1),
            "fast5_pct": round(100 * quick / n, 1),
            "ghost_pct": round(100 * ignored[member] / messages_sent[member], 1) if messages_sent[member] else 0,
        })
    table.sort(key=lambda r: r["median_s"])
    return {"table": table, "replies_received": replies_received}


def swear_stats(msgs):
    member_hits = Counter()
    member_words = defaultdict(Counter)
    word_totals = Counter()
    by_year = Counter()
    total_hits = 0
    for m in msgs:
        found = False
        for w in tokenize(m["content"]):
            if w in CENSOR_WORDS:
                member_words[m["sender"]][w] += 1
                word_totals[w] += 1
                found = True
        if found:
            member_hits[m["sender"]] += 1
            by_year[m["dt"].year] += 1
            total_hits += 1
    return {
        "total_hits": total_hits,
        "member_hits": member_hits,
        "member_words": member_words,
        "word_totals": word_totals,
        "by_year": by_year,
    }


def custom_tracking(msgs, terms):
    if not terms:
        return None
    terms = [t.strip().lower() for t in terms if t.strip()]
    results = {}
    for term in terms:
        per_member = Counter()
        by_year = Counter()
        results[term] = {"count": 0, "per_member": per_member, "by_year": by_year}
    for m in msgs:
        lc = (m["content"] or "").lower()
        if not lc:
            continue
        for term, data in results.items():
            if term in lc:
                data["count"] += 1
                data["per_member"][m["sender"]] += 1
                data["by_year"][m["dt"].year] += 1
    return results


def weird_statements(msgs, top=10):
    lengths = sorted(len(m["content"] or "") for m in msgs if m["content"])
    p99 = lengths[int(0.99 * len(lengths))] if lengths else 0
    scored = []
    for m in msgs:
        text = m["content"] or ""
        if not text or m["mtype"] in ("Call", "Share"):
            continue
        score, reasons = 0, []
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            score += 2
            reasons.append("ALL CAPS")
        if "!!!" in text or "???" in text:
            score += 1
            reasons.append("punctuation spiral")
        if len(text) >= p99 and len(text) >= 200:
            score += 2
            reasons.append("extreme length")
        if m["dt"].hour < 6:
            score += 1
            reasons.append("sent after midnight")
        toks = tokenize(text)
        if any(toks.count(w) >= 4 for w in set(toks)):
            score += 1
            reasons.append("repeated word")
        if score >= 2:
            snippet = text[:160] + ("..." if len(text) > 160 else "")
            scored.append({
                "member": m["sender"], "dt": m["dt"], "score": score,
                "reasons": reasons, "snippet": snippet, "length": len(text),
            })
    scored.sort(key=lambda s: (-s["score"], -s["length"]))
    return scored[:top]


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #

def _bar(fig, ax, labels, values, title, colors=None):
    if colors is None:
        colors = PALETTE
    ax.barh(range(len(labels)), values[::-1], color=[colors[i % len(colors)] for i in range(len(labels))])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i, v in enumerate(values[::-1]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.margins(x=0.15)


def write_charts(msgs, stats, analyses, out_dir, track):
    out_dir.mkdir(parents=True, exist_ok=True)
    theme = {"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#dddddd"}

    def save(fig, name):
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # messages by year
    years = sorted(stats["by_year"])
    with plt.rc_context(theme):
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [stats["by_year"][y] for y in years]
        ax.bar(range(len(years)), vals, color=PALETTE[0])
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
        ax.set_title("Messages per year", fontweight="bold")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        save(fig, "messages_by_year.png")

    # activity by hour
    hours = list(range(24))
    vals = [stats["by_hour"].get(h, 0) for h in hours]
    with plt.rc_context(theme):
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(hours, vals, color=PALETTE[5])
        ax.set_xticks(hours)
        ax.set_title("Messages by hour of day", fontweight="bold")
        save(fig, "activity_by_hour.png")

    # top members
    top_members = stats["member_msgs"].most_common(10)
    with plt.rc_context(theme):
        fig, ax = plt.subplots(figsize=(8, 4))
        _bar(fig, ax, [n for n, _ in top_members], [c for _, c in top_members], "Top members")
        save(fig, "top_members.png")

    # top words
    top_words = stats["words"].most_common(15)
    if top_words:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 5))
            _bar(fig, ax, [w for w, _ in top_words], [c for _, c in top_words], "Top words")
            save(fig, "top_words.png")

    # top emojis
    top_emojis = stats["emojis"].most_common(12)
    if top_emojis:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 5))
            _bar(fig, ax, [emoji_lib.demojize(e).strip(":") for e, _ in top_emojis],
                 [c for _, c in top_emojis], "Top emojis")
            save(fig, "top_emojis.png")

    # yearly recaps: messages per year + top member
    recaps = analyses.get("yearly", {})
    if recaps:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 4))
            ys = sorted(recaps)
            vals = [recaps[y]["total"] for y in ys]
            ax.bar(range(len(ys)), vals, color=[PALETTE[i % len(PALETTE)] for i in range(len(ys))])
            ax.set_xticks(range(len(ys)))
            ax.set_xticklabels(ys, rotation=45, ha="right", fontsize=9)
            ax.set_title("Yearly recap: messages and top member", fontweight="bold")
            for i, y in enumerate(ys):
                ax.text(i, vals[i], recaps[y]["top_member"], ha="center", va="bottom", fontsize=8)
            save(fig, "yearly_recap.png")

    # reactions
    react = analyses.get("reactions", {})
    if react:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_reactors = react["reactor"].most_common(10)
            _bar(fig, ax, [n for n, _ in top_reactors], [c for _, c in top_reactors],
                 "Reactions given (top reactors)")
            save(fig, "reactions_given.png")
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 5))
            pairs = [(m["sender"], m["dt"].strftime("%Y-%m-%d"), len(m["reactions"]), m["content"]) for m in react["top_messages"]]
            labels = [f"{s} {d}: {_shorten(c)}" for s, d, _, c in pairs]
            _bar(fig, ax, labels, [r for _, _, r, _ in pairs], "Most-reacted messages")
            save(fig, "most_reacted.png")

    # response speed
    speed = analyses.get("speed", {})
    if speed["table"]:
        rows = sorted(speed["table"], key=lambda r: r["median_s"])[:10]
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _bar(fig, ax, [r["member"] for r in rows], [max(1, r["median_s"]) for r in rows],
                 "Median time to reply (seconds, log)")
            ax.set_xscale("log")
            save(fig, "response_speed.png")

    # swear stats
    swear = analyses.get("swear", {})
    if swear and swear["total_hits"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top = swear["member_hits"].most_common(10)
            _bar(fig, ax, [n for n, _ in top], [c for _, c in top],
                 "Messages containing swear words (per member)")
            save(fig, "swear_by_member.png")
        ys = sorted(swear["by_year"])
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(ys, [swear["by_year"][y] for y in ys], marker="o", color=PALETTE[3])
            ax.set_xticks(ys)
            ax.set_title("Swearing over time", fontweight="bold")
            save(fig, "swear_over_time.png")

    # custom tracking
    track_data = analyses.get("track")
    if track_data:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 4))
            all_years = sorted({y for t in track_data.values() for y in t["by_year"]})
            width = 0.8 / max(1, len(track_data))
            for i, (term, data) in enumerate(sorted(track_data.items())):
                xs = [all_years.index(y) + (i - (len(track) - 1) / 2) * width for y in data["by_year"]]
                ax.bar(xs, [data["by_year"][y] for y in data["by_year"]],
                       width=width, label=term, color=PALETTE[i % len(PALETTE)])
            ax.set_xticks(range(len(all_years)))
            ax.set_xticklabels(all_years, rotation=45, ha="right")
            ax.set_title("Custom tracked terms per year", fontweight="bold")
            ax.legend(fontsize=8)
            save(fig, "tracked_terms.png")


def _shorten(text, n=40):
    if not text:
        return "(no text)"
    text = text.replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #

def write_summary(title, stats, analyses, track, out_dir, anonymized, dates):
    lines = [f"# {title} flashback", ""]
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                 + (" (names anonymized)" if anonymized else "") + "")
    lines.append(f"- **Period**: {dates[0]} to {dates[-1]}")
    lines.append(f"- **Total messages**: {stats['total']:,}")
    lines.append(f"- **Members**: {len(stats['member_msgs'])}")
    lines.append(f"- **Longest daily streak**: {stats['longest_streak']} days")
    lines.append(f"- **Media (photos/stickers)**: {stats['media']}")
    lines.append(f"- **Links shared**: {stats['links']}")
    lines.append(f"- **Calls**: {stats['calls']} ({int(stats['call_seconds'] // 60)} min)")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Member | Messages | Share |")
    lines.append("|---|---|---|")
    total = max(1, stats["total"])
    for member, count in stats["member_msgs"].most_common(10):
        lines.append(f"| {member} | {count:,} | {100 * count / total:.1f}% |")
    lines.append("")

    lines.append("## Yearly recaps")
    lines.append("")
    for year, recap in analyses["yearly"].items():
        reacted = recap["best_reacted"]
        reacted_note = ""
        if reacted:
            reacted_note = f"; most-reacted: {_shorten(reacted['content'], 30)}"
        lines.append(f"### {year}")
        lines.append(f"- **{recap['total']:,}** messages, **{recap['active_members']}** members active")
        lines.append(f"- Top member: **{recap['top_member']}**")
        lines.append(f"- Top word: **{recap['top_word'][0]}** ({recap['top_word'][1]}x)" if recap["top_word"] else "- Top word: -")
        lines.append(f"- Record day: **{recap['record_day']}** ({recap['record_day_count']} messages){reacted_note}")
        lines.append("")
    lines.append("## Member personalities")
    lines.append("")
    lines.append("| Member | Messages | Words/msg | Peak hour | Night owl % | Signature word | Top emojis |")
    lines.append("|---|---|---|---|---|---|---|")
    for member, p in analyses["personality"].items():
        lines.append(f"| {member} | {p['total_msgs']:,} | {p['avg_words']} | {p['peak_hour']}:00 "
                     f"| {p['night_pct']}% | {p['signature'] or '-'} | {', '.join(p['top_emojis']) or '-'} |")
    lines.append("")

    lines.append("## Reaction dynamics")
    react = analyses["reactions"]
    lines.append("")
    lines.append(f"**{react['total_reactions']}** reactions across **{react['total_reacted']}** messages. "
                 f"Most reactive: **{react['most_reactive']}**.")
    lines.append("")
    lines.append("| Reactor | Reactions given |")
    lines.append("|---|---|")
    for member, count in react["reactor"].most_common(10):
        lines.append(f"| {member} | {count:,} |")
    lines.append("")

    lines.append("## Response speed")
    lines.append("")
    lines.append("Median time to reply, fastest first:")
    lines.append("")
    lines.append("| Member | Replies | Median reply | Replies <5 min | Ghosted % |")
    lines.append("|---|---|---|---|---|")
    for r in analyses["speed"]["table"]:
        lines.append(f"| {r['member']} | {r['replies']} | {r['median_m']} min | {r['fast5_pct']}% | {r['ghost_pct']}% |")
    lines.append("")

    swear = analyses["swear"]
    lines.append("## Swear-word analytics")
    lines.append("")
    if swear["total_hits"]:
        lines.append(f"**{swear['total_hits']}** messages contain profanity.")
        lines.append("")
        lines.append("| Member | Swear messages | Signature swear word |")
        lines.append("|---|---|---|")
        for member, count in swear["member_hits"].most_common(10):
            sig = swear["member_words"][member].most_common(1)[0][0] if swear["member_words"][member] else "-"
            lines.append(f"| {member} | {count} | {sig} |")
    else:
        lines.append("None detected in this thread.")
    lines.append("")

    if track:
        lines.append("## Custom tracked terms")
        lines.append("")
        lines.append("| Term | Total | Top user |")
        lines.append("|---|---|---|")
        for term, data in track.items():
            top = data["per_member"].most_common(1)[0] if data["per_member"] else ("-", 0)
            lines.append(f"| {term} | {data['count']} | {top[0]} ({top[1]}) |")
        lines.append("")

    weird = analyses["weird"]
    lines.append("## Weirdest statements")
    lines.append("")
    if weird:
        for s in weird:
            reasons = ", ".join(s["reasons"])
            lines.append(f"- **{s['member']}** ({s['dt'].strftime('%Y-%m-%d %H:%M')}) "
                         f"- [{reasons}] \"{s['snippet']}\"")
    else:
        lines.append("No weird messages found.")
    lines.append("")

    lines.append("## Charts")
    lines.append("")
    charts = sorted(p.name for p in out_dir.glob("*.png"))
    for c in charts:
        lines.append(f"![{c}]({c})")
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def process_thread(thread_dir, args):
    loaded = load_thread(thread_dir)
    if loaded is None:
        print(f"  [skip] {thread_dir}: no message files")
        return
    title, participants, raw = loaded
    msgs = normalize_messages(raw)
    if not msgs:
        print(f"  [skip] {thread_dir}: no usable messages")
        return
    if args.year:
        msgs = [m for m in msgs if m["dt"].year == args.year]
        if not msgs:
            print(f"  [skip] {thread_dir}: no messages in {args.year}")
            return

    anonymized = False
    if args.anonymize:
        apply_anonymization(msgs, anonymize_map(msgs))
        anonymized = True

    oldest = datetime.fromtimestamp(msgs[0]["ts_ms"] / 1000).strftime("%Y-%m-%d %H:%M")
    newest = datetime.fromtimestamp(msgs[-1]["ts_ms"] / 1000).strftime("%Y-%m-%d %H:%M")
    print(f"\n  Thread: {title}  ({len(msgs):,} messages, {oldest} -> {newest})")

    stats = core_stats(msgs)
    track_terms = [t.strip() for t in args.track.split(",") if t.strip()] if args.track else []
    analyses = {
        "yearly": yearly_recaps(msgs),
        "personality": personalities(msgs, top=args.top),
        "reactions": reaction_stats(msgs, top=args.top),
        "speed": response_speed(msgs, top=args.top),
        "swear": swear_stats(msgs),
        "track": custom_tracking(msgs, track_terms),
        "weird": weird_statements(msgs, top=args.top),
    }

    out_dir = Path(args.output) / _slug(title)
    write_charts(msgs, stats, analyses, out_dir, track_terms)
    write_summary(title, stats, analyses, analyses["track"], out_dir, anonymized,
                  [oldest[:10], newest[:10]])
    print(f"  Wrote output to {out_dir}")
    print(f"  First message: {oldest}  |  Last message: {newest}")
    if not args.anonymize:
        print("  Hint: re-run with --anonymize to strip names for sharing.")


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "thread"


def run(args):
    thread_dirs = find_thread_dirs(args.input)
    if not thread_dirs:
        return 1
    if len(thread_dirs) > 1:
        print(f"Found {len(thread_dirs)} threads:")
        for d in thread_dirs[:20]:
            print(f"  - {d}")
        print("Processing all threads...")
    for d in thread_dirs:
        process_thread(d, args)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="analyze_chat.py",
        description=(
            "Analyze a Facebook Messenger export and generate flashback analytics: "
            "yearly recaps, member personalities, reaction dynamics, response-speed "
            "leaderboards, swear-word stats, custom term tracking, and a 'weirdest "
            "statements' highlight reel. Runs 100% locally."
        ),
    )
    parser.add_argument("--input", "-i", default="data",
                        help="Chat export folder (default: data/)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help="Where to write charts and summary.md (default: output/)")
    parser.add_argument("--anonymize", action="store_true",
                        help="Replace member names with Person A, Person B, ... in every report")
    parser.add_argument("--track", default="",
                        help='Comma-separated words/phrases to count and chart, e.g. --track "lol, bro"')
    parser.add_argument("--year", type=int,
                        help="Limit the analysis to a single year, e.g. --year 2017")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of entries in leaderboards (default: 10)")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
