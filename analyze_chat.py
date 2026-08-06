#!/usr/bin/env python3
"""chat-flashback: turn a Messenger export into a report about a group chat.

Parses a Facebook Messenger JSON export (or a WhatsApp/Telegram folder), computes
the analytics, and writes charts plus a summary.md. Everything runs locally.
"""

import argparse
import base64
import html as html_lib
import io
import json
import math
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import emoji as emoji_lib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from better_profanity import Profanity

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _Vader
    _VADER = _Vader()
except Exception:
    _VADER = None

DEFAULT_OUTPUT = "output"
# Analyses that can be turned off with --skip. "jokes" dominates peak memory
# (it counts every 2-4 word phrase in the chat) and "sentiment" is the slowest,
# so those two are what to drop on a very large chat or a small machine.
SKIPPABLE = ("jokes", "sentiment", "wordcloud", "topics")
REPLY_WINDOW_SECONDS = 60 * 60
CONVERSATION_WINDOW_SECONDS = 30 * 60
MESSAGE_FILE_RE = re.compile(r"^message_\d+\.json$")

KNOWN_MESSAGE_KEYS = {
    "id", "sender_name", "timestamp_ms", "timestamp", "content", "type", "reactions",
    "share", "photos", "sticker", "gifs", "videos", "audio_files", "files", "polls",
    "call_duration", "reply_to_message_id", "is_unsent", "is_taken_down", "text",
}

KNOWN_TYPES = {
    "Generic", "Call", "Share", "Subscribe", "Unsubscribe", "Game", "Poll", "Plan",
    "Money", "Unsend", "GroupPoll", "ShareSticker", "Payment", "Checkout", "VideoCall",
    "MultiwayCall", "MysteryChat", "Pin", "GroupCreate", "GroupUpdate", "Reaction",
    "GroupNameChange", "GroupAvatarChange", "FriendRequest", "ProfilePic",
    "EventReminder", "AdminChange", "Archived", "Encrypted", "ThreadIcon",
    "text", "share", "sticker", "gif", "video", "audio", "file", "call", "system",
}

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
    """Fix Messenger's double-encoded unicode (literal \\u00xx escapes of UTF-8 bytes).

    A double-encoded emoji arrives as the text "\\u00f0\\u009f\\u0091\\u008d".
    Convert each escape to its byte value, then decode the bytes as UTF-8.
    """
    if not text or "\\u00" not in text:
        return text
    fixed = re.sub(r"\\u00([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)
    if any(ord(c) > 0xFF for c in fixed):
        return text
    try:
        return fixed.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
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
    seen = set()
    unique = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        key = m.get("id")
        if key is None:
            key = (m.get("sender_name"), m.get("timestamp_ms") or m.get("timestamp"),
                   m.get("content"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return title, participants, unique


def _parse_tz(tz):
    tz = tz.strip()
    m = re.match(r"^([+-])(\d{1,2}):?(\d{2})?$", tz)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours = int(m.group(2))
        minutes = int(m.group(3) or 0)
        if hours > 14 or minutes > 59:
            raise ValueError(f"invalid timezone offset {tz!r}")
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz)
    except Exception:
        raise ValueError(
            f"invalid timezone {tz!r}; use +HH:MM / -HH:MM or an IANA name like "
            "America/New_York")


def _local_dt(ts_ms, tzinfo):
    if tzinfo is None:
        return datetime.fromtimestamp(ts_ms / 1000)
    return datetime.fromtimestamp(ts_ms / 1000, tzinfo).replace(tzinfo=None)


def normalize_messages(raw, tz=None, consume=False):
    """Turn raw export dicts into the normalized messages the analyses use.

    With consume=True the input list is emptied as it is read, so the raw and
    normalized copies of a large chat never both sit in memory. Entries are
    taken from the end, which is why the result is sorted at the end anyway.
    """
    tzinfo = _parse_tz(tz) if tz else None
    msgs = []
    for i, m in enumerate(raw):
        if consume:
            raw[i] = None       # drop the raw dict as soon as it is converted
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
        photo_uris = [p.get("uri") for p in (m.get("photos") or []) if isinstance(p, dict) and p.get("uri")]
        gif_uris = [g.get("uri") for g in (m.get("gifs") or []) if isinstance(g, dict) and g.get("uri")]
        video_uris = [v.get("uri") for v in (m.get("videos") or []) if isinstance(v, dict) and v.get("uri")]
        audio_uris = [a.get("uri") for a in (m.get("audio_files") or []) if isinstance(a, dict) and a.get("uri")]
        file_uris = [f.get("uri") for f in (m.get("files") or []) if isinstance(f, dict) and f.get("uri")]
        file_names = [f.get("name") or f.get("uri") for f in (m.get("files") or [])
                      if isinstance(f, dict) and (f.get("name") or f.get("uri"))]
        poll_question = None
        for poll in m.get("polls") or []:
            if isinstance(poll, dict) and poll.get("question"):
                poll_question = poll["question"]
                break
        if content is None and poll_question:
            content = poll_question
        has_media = bool(photo_uris or gif_uris or video_uris or audio_uris
                         or file_uris or file_names or m.get("sticker"))
        msgs.append({
            "id": m.get("id"),
            "sender": sender,
            "ts_ms": ts_ms,
            "dt": _local_dt(ts_ms, tzinfo),
            "content": content,
            "mtype": m.get("type", "Generic"),
            "reactions": reactions,
            "has_photo": bool(m.get("photos")),
            "has_sticker": bool(m.get("sticker")),
            "has_gif": bool(m.get("gifs")),
            "has_video": bool(m.get("videos")),
            "has_audio": bool(m.get("audio_files")),
            "has_file": bool(file_uris or file_names),
            "has_media": has_media,
            "photo_uris": photo_uris,
            "gif_uris": gif_uris,
            "video_uris": video_uris,
            "audio_uris": audio_uris,
            "file_uris": file_uris,
            "file_names": file_names,
            "poll_question": poll_question,
            "link": (share or {}).get("link"),
            "call_duration": m.get("call_duration"),
            "reply_to": m.get("reply_to_message_id"),
            "is_unsent": bool(m.get("is_unsent")),
            "is_taken_down": bool(m.get("is_taken_down")),
        })
    msgs.sort(key=lambda x: x["ts_ms"])
    return msgs


def anonymize_map(msgs):
    counts = Counter(m["sender"] for m in msgs)
    names = sorted(counts, key=lambda n: (-counts[n], msgs[0]["sender"] != n))
    return {n: f"Person {chr(ord('A') + i)}" for i, n in enumerate(names)}


def apply_anonymization(msgs, mapping):
    """Replace every member name in senders, reactors and message text.

    Names are matched longest-first in a single pass: replacing them one at a
    time in count order lets a short name rewrite part of a longer one that
    contains it ("Ann" inside "Ann Smith" would leave the surname behind), and
    a single pass means a label can never be rewritten by a later name.
    """
    names = sorted(mapping, key=len, reverse=True)
    combined = (re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b",
                           re.IGNORECASE) if names else None)
    lookup = {n.lower(): label for n, label in mapping.items()}
    for m in msgs:
        m["sender"] = mapping.get(m["sender"], "Person ?")
        m["reactions"] = [(mapping.get(a, "Person ?"), r) for a, r in m["reactions"]]
        content = m["content"]
        if content and combined is not None:
            m["content"] = combined.sub(
                lambda mo: lookup.get(mo.group(1).lower(), "Person ?"), content)
    return msgs


# --------------------------------------------------------------------------- #
# Core stats                                                                  #
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    """Split into words, keeping digits that sit inside a word.

    "covid19" and "b2b" stay whole; bare numbers like "2020" are dropped, since
    they are dates and counts rather than vocabulary.
    """
    return [w for w in _WORD_RE.findall((text or "").lower())
            if any(c.isalpha() for c in w)]


def split_emojis(text):
    return [c for c in (text or "") if emoji_lib.is_emoji(c)]


def add_derived_fields(msgs):
    """Tokenize and scan for emojis once per message, up front.

    Nine analyses tokenize the same text and four scan it for emojis. Doing
    that per analysis re-runs the same regex, and the same per-character emoji
    lookup, over the whole chat nine times.

    Tokens are interned, so a long chat holds pointers into its own vocabulary
    instead of millions of duplicate strings, and messages without text share
    the empty tuple. Must run after any anonymization, which rewrites content.
    """
    for m in msgs:
        content = m["content"]
        if content:
            m["tokens"] = tuple(sys.intern(w) for w in tokenize(content))
            m["emojis"] = tuple(sys.intern(e) for e in split_emojis(content))
        else:
            m["tokens"] = ()
            m["emojis"] = ()
    return msgs


def _tokens(m):
    """Cached tokens, falling back for callers that skipped enrichment."""
    cached = m.get("tokens")
    return cached if cached is not None else tuple(tokenize(m["content"]))


def _emojis(m):
    cached = m.get("emojis")
    return cached if cached is not None else tuple(split_emojis(m["content"]))


def _median(sorted_values):
    n = len(sorted_values)
    if not n:
        return None
    if n % 2:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2


def _sum_counters(counters):
    """Add Counters together.

    A dict comprehension over several Counters silently keeps only the last
    value for a repeated key, so totals have to be accumulated.
    """
    total = Counter()
    for c in counters:
        total.update(c)
    return total


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
    total_words = 0
    for m in msgs:
        member_msgs[m["sender"]] += 1
        by_hour[m["dt"].hour] += 1
        by_weekday[m["dt"].weekday()] += 1
        by_month[m["dt"].month] += 1
        by_year[m["dt"].year] += 1
        by_day[m["dt"].date()] += 1
        toks = _tokens(m)
        total_words += len(toks)
        for w in toks:
            if w not in STOPWORDS and len(w) > 1:
                words[w] += 1
                per_member_words[m["sender"]][w] += 1
        for e in _emojis(m):
            emojis[e] += 1
            per_member_emojis[m["sender"]][e] += 1
    media = sum(1 for m in msgs if m["has_media"])
    links = sum(1 for m in msgs if m["link"])
    calls = sum(1 for m in msgs if m["mtype"] == "Call")
    call_seconds = sum(m["call_duration"] or 0 for m in msgs if m["mtype"] == "Call")
    return {
        "total": len(msgs),
        "total_words": total_words,
        "active_days": len(by_day),
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
            words.update(w for w in _tokens(m) if w not in STOPWORDS and len(w) > 1)
            emojis.update(_emojis(m))
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
        for w in _tokens(m):
            if w not in STOPWORDS and len(w) > 1:
                per_member_words[m["sender"]][w] += 1
                word_totals[w] += 1
        for e in _emojis(m):
            per_member_emojis[m["sender"]][e] += 1
    for member, group in per_member.items():
        total_msgs = len(group)
        total_words = sum(len(_tokens(m)) for m in group)
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
    """Reply latency per member, plus how often each member's turn went unanswered.

    Ghosting is measured per *turn* (a run of consecutive messages by one
    member), not per message: only the end of a run can be replied to, so
    dividing by every message sent would count messages that were never
    candidates for a reply. The final run of the export is ignored, since
    nobody had the chance to answer it.
    """
    reply_seconds = defaultdict(list)
    replies_received = Counter()
    messages_sent = Counter(m["sender"] for m in msgs)
    turns = Counter()
    ignored = Counter()
    i, n = 0, len(msgs)
    while i < n:
        sender = msgs[i]["sender"]
        j = i + 1
        while j < n and msgs[j]["sender"] == sender:
            j += 1
        if j < n:
            turns[sender] += 1
            gap = (msgs[j]["ts_ms"] - msgs[j - 1]["ts_ms"]) / 1000
            if 0 <= gap <= REPLY_WINDOW_SECONDS:
                reply_seconds[msgs[j]["sender"]].append(gap)
                replies_received[sender] += 1
            else:
                ignored[sender] += 1
        i = j
    table = []
    for member in sorted(messages_sent):
        gaps = sorted(reply_seconds.get(member, []))
        replies = len(gaps)
        med = _median(gaps)
        quick = sum(1 for g in gaps if g <= 300)
        member_turns = turns[member]
        table.append({
            "member": member,
            "replies": replies,
            "median_s": med,
            "median_m": round(med / 60, 1) if med is not None else None,
            "fast5_pct": round(100 * quick / replies, 1) if replies else None,
            "turns": member_turns,
            "ghost_pct": (round(100 * ignored[member] / member_turns, 1)
                          if member_turns else None),
        })
    # Fastest first; members who never replied have no median and sort last.
    table.sort(key=lambda r: (r["median_s"] is None, r["median_s"] or 0))
    return {"table": table, "replies_received": replies_received}


def _fastest_replier(speed):
    """First row of the response-speed table that actually has a median."""
    for row in (speed or {}).get("table") or []:
        if row["median_s"] is not None:
            return row
    return None


def swear_stats(msgs):
    member_hits = Counter()
    member_words = defaultdict(Counter)
    word_totals = Counter()
    by_year = Counter()
    total_hits = 0
    for m in msgs:
        found = False
        for w in _tokens(m):
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
        toks = _tokens(m)
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


def _unwrap_link(link):
    try:
        parts = urlsplit(link)
        if parts.netloc in ("l.facebook.com", "l.php"):
            target = (parse_qs(parts.query).get("u") or [None])[0]
            if target:
                return unquote(target)
    except ValueError:
        pass
    return link


def links_domains(msgs, top=10):
    domains = Counter()
    links = Counter()
    for m in msgs:
        link = m.get("link")
        if not link:
            continue
        url = _unwrap_link(link)
        try:
            host = urlsplit(url).netloc.lower() or "direct"
        except ValueError:
            host = "direct"
        if "lookaside.fbsbx.com" in url or "scontent" in url:
            continue
        if host.endswith("facebook.com") and "/photo" in url:
            continue
        host = host.replace("www.", "")
        domains[host] += 1
        links[url] += 1
    return {"domains": domains, "links": links}


def media_leaderboard(msgs):
    photos = Counter()
    stickers = Counter()
    gifs = Counter()
    videos = Counter()
    audio = Counter()
    files = Counter()
    for m in msgs:
        if m["has_photo"]:
            photos[m["sender"]] += 1
        if m["has_sticker"]:
            stickers[m["sender"]] += 1
        if m["has_gif"]:
            gifs[m["sender"]] += 1
        if m["has_video"]:
            videos[m["sender"]] += 1
        if m["has_audio"]:
            audio[m["sender"]] += 1
        if m["has_file"]:
            files[m["sender"]] += 1
    return {"photos": photos, "stickers": stickers, "gifs": gifs,
            "videos": videos, "audio": audio, "files": files}


def length_trends(msgs):
    by_year = defaultdict(list)
    for m in msgs:
        if m["content"]:
            by_year[m["dt"].year].append(len(m["content"]))
    return {y: {"n": len(vals), "avg_chars": round(sum(vals) / len(vals), 1),
                "max_chars": max(vals)} for y, vals in sorted(by_year.items())}


def word_trends(msgs, top=5):
    totals = Counter()
    per_year_word = defaultdict(Counter)
    for m in msgs:
        y = m["dt"].year
        for w in _tokens(m):
            if w not in STOPWORDS and len(w) > 1:
                totals[w] += 1
                per_year_word[y][w] += 1
    top_words = [w for w, _ in totals.most_common(top)]
    return {w: {y: per_year_word[y].get(w, 0) for y in sorted(per_year_word)} for w in top_words}


def conversation_starters(msgs):
    runs = []
    current = []
    prev_ts = None
    for m in msgs:
        if prev_ts is not None and (m["ts_ms"] - prev_ts) / 1000 > CONVERSATION_WINDOW_SECONDS:
            if current:
                runs.append(current)
            current = [m]
        else:
            current.append(m)
        prev_ts = m["ts_ms"]
    if current:
        runs.append(current)
    starter = Counter(run[0]["sender"] for run in runs)
    longest = max(runs, key=len) if runs else []
    return {"starters": starter, "conversation_count": len(runs),
            "longest_run": longest, "longest_run_len": len(longest)}


def reply_chains(msgs, top=5):
    by_id = {m["id"]: m for m in msgs if m.get("id") is not None}
    if not by_id:
        return None
    referenced = {m.get("reply_to") for m in msgs if m.get("reply_to") is not None}
    terminals = [m for m in msgs
                 if m.get("reply_to") is not None and m.get("id") not in referenced]
    chain_list = []
    for m in terminals:
        chain = []
        cur = m
        seen = set()
        while cur is not None and cur.get("id") not in seen:
            seen.add(cur.get("id"))
            chain.append(cur)
            cur = by_id.get(cur.get("reply_to"))
        if len(chain) >= 2:
            chain_list.append(chain)
    chain_list.sort(key=lambda c: (-len(c), c[-1]["ts_ms"]))
    count = sum(len(c) for c in chain_list)
    return {"count": count, "top_chains": chain_list[:top]}


def ghosting(msgs):
    per_member_days = defaultdict(set)
    for m in msgs:
        per_member_days[m["sender"]].add(m["dt"].date())
    gaps = {}
    for member, days in per_member_days.items():
        days = sorted(days)
        max_gap = 0
        for a, b in zip(days, days[1:]):
            max_gap = max(max_gap, (b - a).days - 1)
        gaps[member] = max_gap
    return gaps


def extremes(msgs):
    if not msgs:
        return None
    longest = max(msgs, key=lambda m: len(m["content"] or ""))
    shortest = min((m for m in msgs if m["content"] and m["content"].strip()),
                   key=lambda m: len(m["content"]), default=None)
    most_reacted = max(msgs, key=lambda m: len(m["reactions"]), default=None)
    monthly = Counter()
    for m in msgs:
        monthly[m["dt"].strftime("%Y-%m")] += 1
    day_counts = Counter(m["dt"].date() for m in msgs)
    record_day, record_day_count = day_counts.most_common(1)[0]
    return {"longest": longest, "shortest": shortest, "most_reacted": most_reacted,
            "monthly": monthly, "record_day": record_day, "record_day_count": record_day_count}


def sentiment_analysis(msgs):
    if _VADER is None:
        return None
    cache = {}
    member_scores = defaultdict(list)
    year_scores = defaultdict(list)
    for m in msgs:
        text = m["content"]
        if not text:
            continue
        compound = cache.get(text)
        if compound is None:
            compound = _VADER.polarity_scores(text)["compound"]
            cache[text] = compound
            if len(cache) > 250_000:
                cache.clear()
        member_scores[m["sender"]].append(compound)
        year_scores[m["dt"].year].append(compound)

    def avg(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "per_member": {m: avg(v) for m, v in member_scores.items()},
        "per_year": {y: avg(v) for y, v in sorted(year_scores.items())},
        "messages_scored": sum(len(v) for v in member_scores.values()),
    }


def activity_heatmap(msgs):
    by_day = Counter()
    for m in msgs:
        by_day[m["dt"].date()] += 1
    if not by_day:
        return None
    start = min(by_day)
    end = max(by_day)
    first_monday = start - timedelta(days=start.weekday())
    last_monday = end - timedelta(days=end.weekday())
    weeks = []
    week_starts = []
    d = first_monday
    while d <= last_monday:
        weeks.append([by_day.get(d + timedelta(days=i), 0) for i in range(7)])
        week_starts.append(d)
        d += timedelta(days=7)
    return {"weeks": weeks, "week_starts": week_starts,
            "max_day": max(by_day.values()), "active_days": len(by_day)}


def pace_trends(msgs):
    by_day = Counter()
    calls_by_month = Counter()
    media_by_year = Counter()
    for m in msgs:
        by_day[m["dt"].date()] += 1
        if m["mtype"] == "Call":
            calls_by_month[m["dt"].strftime("%Y-%m")] += 1
        if m["has_media"]:
            media_by_year[m["dt"].year] += 1
    days = sorted(by_day)
    counts = [by_day[d] for d in days]
    rolling = []
    for i in range(len(days)):
        lo = max(0, i - 29)
        rolling.append(round(sum(counts[lo:i + 1]) / (i - lo + 1), 2))
    return {"days": [d.strftime("%Y-%m-%d") for d in days], "counts": counts,
            "rolling": rolling, "calls_by_month": calls_by_month,
            "media_by_year": media_by_year}


def pair_matrices(msgs):
    members = sorted({m["sender"] for m in msgs})
    reply = defaultdict(lambda: defaultdict(int))
    for prev, cur in zip(msgs, msgs[1:]):
        if cur["sender"] == prev["sender"]:
            continue
        gap = (cur["ts_ms"] - prev["ts_ms"]) / 1000
        if 0 < gap <= REPLY_WINDOW_SECONDS:
            reply[cur["sender"]][prev["sender"]] += 1
    reaction = defaultdict(lambda: defaultdict(int))
    for m in msgs:
        if not m["reactions"]:
            continue
        target = m["sender"]
        for actor, _ in m["reactions"]:
            reaction[actor][target] += 1
    return {"members": members, "reply": reply, "reaction": reaction}


def hourly_radar(msgs):
    per_member = defaultdict(lambda: [0] * 24)
    for m in msgs:
        per_member[m["sender"]][m["dt"].hour] += 1
    return {member: hours for member, hours in per_member.items()}


def word_cloud_data(msgs):
    overall = Counter()
    per_member = defaultdict(Counter)
    for m in msgs:
        for w in _tokens(m):
            if w not in STOPWORDS and len(w) > 1:
                overall[w] += 1
                per_member[m["sender"]][w] += 1
    return {"overall": overall, "per_member": per_member}


def monologues(msgs):
    runs = []
    current = []
    prev = None
    for m in msgs:
        same_sender = prev is not None and m["sender"] == prev["sender"]
        in_window = prev is not None and (m["ts_ms"] - prev["ts_ms"]) / 1000 <= CONVERSATION_WINDOW_SECONDS
        if same_sender and in_window:
            current.append(m)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [m]
        prev = m
    if len(current) >= 2:
        runs.append(current)
    per_member = Counter()
    for run in runs:
        if len(run) > per_member[run[0]["sender"]]:
            per_member[run[0]["sender"]] = len(run)
    longest = max(runs, key=len) if runs else []
    return {"runs": runs, "per_member_longest": per_member,
            "longest_run": longest, "longest_run_len": len(longest),
            "email_moments": sum(1 for r in runs if len(r) >= 4)}


def unsent_stats(msgs):
    return Counter(m["sender"] for m in msgs if m["is_unsent"])


def taken_down_stats(msgs):
    return Counter(m["sender"] for m in msgs if m["is_taken_down"])


def emoji_stats(msgs):
    per_member = defaultdict(Counter)
    per_year = defaultdict(Counter)
    member_total = Counter(m["sender"] for m in msgs)
    for m in msgs:
        emojis = _emojis(m)
        if not emojis:
            continue
        per_member[m["sender"]].update(emojis)
        per_year[m["dt"].year].update(emojis)
    return {
        "total_emojis": sum(sum(c.values()) for c in per_member.values()),
        "per_member": dict(per_member),
        "per_year": {y: c for y, c in sorted(per_year.items())},
        "emojis_per_100": {m: round(100 * sum(c.values()) / member_total[m], 1)
                           for m, c in per_member.items() if member_total[m]},
    }


def question_stats(msgs):
    def is_question(m):
        c = (m["content"] or "").strip()
        return bool(c) and c.endswith("?")

    n = len(msgs)
    next_other = [None] * n
    i = 0
    while i < n:
        sender = msgs[i]["sender"]
        j = i + 1
        while j < n and msgs[j]["sender"] == sender:
            j += 1
        if j < n:
            # Gaps shrink as k approaches j, so an early message being out of
            # the window says nothing about the later ones: skip, do not stop.
            for k in range(i, j):
                if (msgs[j]["ts_ms"] - msgs[k]["ts_ms"]) / 1000 <= REPLY_WINDOW_SECONDS:
                    next_other[k] = j
        i = j

    asked = Counter()
    answered_q = Counter()
    responses = Counter()
    answer_time = defaultdict(list)
    total_asked = 0
    total_answered = 0
    for idx, m in enumerate(msgs):
        if not is_question(m):
            continue
        total_asked += 1
        asked[m["sender"]] += 1
        j = next_other[idx]
        if j is not None:
            total_answered += 1
            answered_q[m["sender"]] += 1
            responder = msgs[j]["sender"]
            responses[responder] += 1
            answer_time[responder].append((msgs[j]["ts_ms"] - m["ts_ms"]) / 1000)

    table = []
    for member in sorted(set(asked) | set(responses)):
        a = asked[member]
        gaps = sorted(answer_time[member])
        med = _median(gaps)
        table.append({
            "member": member,
            "asked": a,
            "answered": answered_q[member],
            "answer_pct": round(100 * answered_q[member] / a, 1) if a else None,
            "responses_given": responses[member],
            "median_m": round(med / 60, 1) if med is not None else None,
        })
    # Name breaks ties so equal rows keep the same order from run to run.
    table.sort(key=lambda r: (-r["asked"], r["member"]))
    return {"table": table, "total_questions": total_asked,
            "total_answered": total_answered,
            "unanswered_count": total_asked - total_answered}


def topic_words(msgs, top=6):
    """Words that characterise each year, scored with tf-idf over years.

    Each *year* is one document, so the document frequency of a word is the
    number of years it shows up in. Counting messages instead makes the idf
    term go negative for anything common, which inverts the ranking and
    surfaces one-off typos as a year's topics.
    """
    by_year = defaultdict(Counter)
    year_totals = Counter()
    years_with_word = defaultdict(set)
    for m in msgs:
        year = m["dt"].year
        for w in _tokens(m):
            if w in STOPWORDS or len(w) <= 2:
                continue
            by_year[year][w] += 1
            year_totals[year] += 1
            years_with_word[w].add(year)
    n_years = max(1, len(by_year))
    result = {}
    for year, counter in sorted(by_year.items()):
        tf_total = max(1, year_totals[year])
        scored = []
        for w, c in counter.items():
            tf = c / tf_total
            # 1.0 for a word used every year, rising as it narrows to fewer.
            idf = math.log(n_years / len(years_with_word[w])) + 1.0
            scored.append((tf * idf, w, c))
        scored.sort(reverse=True)
        result[year] = [{"word": w, "score": round(s, 4), "count": c}
                        for s, w, c in scored[:top]]
    return {"by_year": result, "years": sorted(by_year)}


def inside_jokes(msgs, min_count=4, min_years=2, min_members=2, top=12):
    """Repeated phrases said by enough people over enough years to be a joke.

    Done in two passes to keep memory bounded on long chats. Tracking every
    2-, 3- and 4-gram with its member/year sets and an example string costs
    roughly 17 kB per message, which is gigabytes for a decade of group chat.

    Pass one counts bigrams alone (ints, no payload). A phrase can never occur
    more often than any bigram inside it, so a 3- or 4-gram can only reach
    min_count if all of its bigrams already did. Pass two therefore builds the
    full records only for phrases that survive that test, which is exact -- no
    qualifying phrase can be pruned -- while discarding the long tail of
    said-once phrases that dominates the count.
    """
    def phrase_words(m):
        return [w for w in _tokens(m) if w not in STOPWORDS and len(w) > 2]

    def ngrams(words, n):
        return [" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]

    bigram_counts = Counter()
    for m in msgs:
        words = phrase_words(m)
        if len(words) >= 2:
            bigram_counts.update(set(ngrams(words, 2)))
    frequent = {g for g, c in bigram_counts.items() if c >= min_count}
    bigram_counts.clear()
    if not frequent:
        return {"jokes": [], "total_candidates": 0}

    phrase = defaultdict(lambda: {"count": 0, "members": set(), "years": set(), "example": None})
    for m in msgs:
        words = phrase_words(m)
        if len(words) < 2:
            continue
        found = set()
        for n in (2, 3, 4):
            for i in range(max(0, len(words) - n + 1)):
                parts = words[i:i + n]
                if not all(" ".join(parts[k:k + 2]) in frequent for k in range(n - 1)):
                    continue
                g = " ".join(parts)
                if g in found:
                    continue
                found.add(g)
                info = phrase[g]
                info["count"] += 1
                info["members"].add(m["sender"])
                info["years"].add(m["dt"].year)
                if info["example"] is None:
                    info["example"] = m["content"]
    jokes = []
    for phrase_text, info in phrase.items():
        if (info["count"] >= min_count and len(info["members"]) >= min_members
                and len(info["years"]) >= min_years):
            ex = info["example"]
            if ex and len(ex) > 70:
                ex = ex[:70] + "..."
            jokes.append({
                "phrase": phrase_text, "count": info["count"],
                "members": sorted(info["members"]), "years": sorted(info["years"]),
                "example": ex,
            })
    jokes.sort(key=lambda j: (-j["count"], -len(j["years"])))
    return {"jokes": jokes[:top], "total_candidates": len(phrase)}


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #

def _bar(fig, ax, labels, values, title, colors=None):
    if colors is None:
        colors = PALETTE
    ax.barh(range(len(labels)), values[::-1], color=[colors[i % len(colors)] for i in range(len(labels))])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(["\n".join(textwrap.wrap(str(l), 40)) for l in labels[::-1]], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i, v in enumerate(values[::-1]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.margins(x=0.15)


def write_charts(msgs, stats, analyses, out_dir, track, top=10):
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

    # activity by weekday
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    vals = [stats["by_weekday"].get(d, 0) for d in range(7)]
    with plt.rc_context(theme):
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(range(7), vals, color=PALETTE[2])
        ax.set_xticks(range(7))
        ax.set_xticklabels(weekdays)
        ax.set_title("Messages by weekday", fontweight="bold")
        save(fig, "activity_by_weekday.png")

    # top members
    top_members = stats["member_msgs"].most_common(top)
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
            top_reactors = react["reactor"].most_common(top)
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
    speed = analyses.get("speed") or {}
    speed_rows = [r for r in speed.get("table") or [] if r["median_s"] is not None]
    if speed_rows:
        rows = sorted(speed_rows, key=lambda r: r["median_s"])[:top]
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
            top_swearers = swear["member_hits"].most_common(top)
            _bar(fig, ax, [n for n, _ in top_swearers], [c for _, c in top_swearers],
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
            n_terms = max(1, len(track_data))
            width = 0.8 / n_terms
            for i, (term, data) in enumerate(sorted(track_data.items())):
                xs = [all_years.index(y) + (i - (n_terms - 1) / 2) * width for y in data["by_year"]]
                ax.bar(xs, [data["by_year"][y] for y in data["by_year"]],
                       width=width, label=term, color=PALETTE[i % len(PALETTE)])
            ax.set_xticks(range(len(all_years)))
            ax.set_xticklabels(all_years, rotation=45, ha="right")
            ax.set_title("Custom tracked terms per year", fontweight="bold")
            ax.legend(fontsize=8)
            save(fig, "tracked_terms.png")

    # top domains
    ld = analyses.get("links_domains", {})
    if ld and ld["domains"]:
        top_domains = ld["domains"].most_common(top)
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            _bar(fig, ax, [d for d, _ in top_domains], [c for _, c in top_domains],
                 "Top domains shared")
            save(fig, "top_domains.png")

    # media leaderboard
    media = analyses.get("media", {})
    if media and any(media[k] for k in ("photos", "stickers", "gifs", "videos", "audio", "files")):
        members = sorted(set().union(*[set(media[k]) for k in
                                       ("photos", "stickers", "gifs", "videos", "audio", "files")]))
        labels = ["Photos", "Stickers", "GIFs", "Videos", "Audio", "Files"]
        keys = ["photos", "stickers", "gifs", "videos", "audio", "files"]
        series = [[media[k].get(m, 0) for m in members] for k in keys]
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 5))
            bottom = [0] * len(members)
            for i, (label, vals) in enumerate(zip(labels, series)):
                ax.bar(members, vals, bottom=bottom, label=label, color=PALETTE[i % len(PALETTE)])
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.tick_params(axis="x", labelrotation=30, labelsize=9)
            ax.set_title("Media sent (per member)", fontweight="bold")
            ax.legend(fontsize=8)
            save(fig, "media_leaderboard.png")

    # message-length trends
    lengths = analyses.get("length_trends", {})
    if lengths:
        ys = list(lengths)
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(ys, [lengths[y]["avg_chars"] for y in ys], marker="o", color=PALETTE[0])
            ax.set_xticks(ys)
            ax.set_title("Average message length (chars) per year", fontweight="bold")
            save(fig, "length_trends.png")

    # word trends over time (fallback top words)
    wt = analyses.get("word_trends", {})
    if wt:
        # Plot against tick positions, not the year values themselves: mixing
        # the two puts the data at x=2017.. while the ticks sit at x=0..n.
        years = list(next(iter(wt.values())))
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, (word, series) in enumerate(sorted(wt.items())):
                ax.plot(range(len(years)), [series.get(y, 0) for y in years],
                        marker="o", label=word, color=PALETTE[i % len(PALETTE)])
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
            ax.set_title("Top words over time", fontweight="bold")
            ax.legend(fontsize=8)
            save(fig, "word_trends.png")

    # conversation starters
    conv = analyses.get("conversations", {})
    if conv and conv["starters"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_starters = conv["starters"].most_common(top)
            _bar(fig, ax, [n for n, _ in top_starters], [c for _, c in top_starters],
                 "Conversations started (30-min gap)")
            save(fig, "conversation_starters.png")

    # reply chains
    chains = analyses.get("reply_chains")
    if chains and chains["top_chains"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [f"#{i+1} ({len(c)} msgs)" for i, c in enumerate(chains["top_chains"])]
            _bar(fig, ax, labels, [len(c) for c in chains["top_chains"]], "Longest reply chains")
            save(fig, "reply_chains.png")

    # ghosting
    ghosts = analyses.get("ghosting", {})
    if ghosts:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_ghosts = sorted(ghosts.items(), key=lambda kv: -kv[1])[:top]
            _bar(fig, ax, [n for n, _ in top_ghosts], [c for _, c in top_ghosts],
                 "Longest silent streak (days)")
            save(fig, "ghosting.png")

    # monthly timeline
    ex = analyses.get("extremes")
    if ex and ex["monthly"]:
        months = sorted(ex["monthly"])
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(11, 3.5))
            vals = [ex["monthly"][mo] for mo in months]
            ax.bar(range(len(months)), vals, color=PALETTE[4])
            ax.set_xticks(range(0, len(months), max(1, len(months) // 12)))
            ax.set_xticklabels([months[i][:7] for i in range(0, len(months), max(1, len(months) // 12))],
                               rotation=45, ha="right", fontsize=8)
            ax.set_title("Messages per month", fontweight="bold")
            peak = max(vals)
            rd_month = ex["record_day"].strftime("%Y-%m")
            if rd_month in months:
                idx = months.index(rd_month)
                ax.annotate("record day", xy=(idx, ex["monthly"][rd_month]),
                            xytext=(idx, peak * 1.05), ha="center", fontsize=8,
                            color=PALETTE[3], fontweight="bold")
            mr = ex.get("most_reacted")
            if mr:
                mr_month = mr["dt"].strftime("%Y-%m")
                if mr_month in months:
                    idx = months.index(mr_month)
                    ax.annotate("most reacted", xy=(idx, ex["monthly"][mr_month]),
                                xytext=(idx, -peak * 0.08), ha="center", fontsize=8,
                                color=PALETTE[0], fontweight="bold")
            save(fig, "monthly_timeline.png")

    # sentiment
    senti = analyses.get("sentiment")
    if senti:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_senti = sorted(senti["per_member"].items(), key=lambda kv: -kv[1])[:top]
            _bar(fig, ax, [n for n, _ in top_senti], [round(c * 100) for _, c in top_senti],
                 "Average sentiment (VADER x100, higher = happier)")
            save(fig, "sentiment_per_member.png")
        ys = list(senti["per_year"])
        if ys:
            with plt.rc_context(theme):
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(ys, [senti["per_year"][y] * 100 for y in ys], marker="o", color=PALETTE[1])
                ax.axhline(0, color="#999999", linewidth=0.8)
                ax.set_xticks(ys)
                ax.set_title("Average sentiment per year (VADER x100)", fontweight="bold")
                save(fig, "sentiment_over_time.png")

    # activity heatmap (GitHub-style calendar)
    hm = analyses.get("heatmap")
    if hm:
        grid = np.array(hm["weeks"]).T
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(max(6, len(hm["weeks"]) * 0.13), 2.8))
            im = ax.imshow(grid, aspect="auto", cmap="Blues",
                           vmin=0, vmax=max(1, hm["max_day"]))
            ax.set_yticks(range(7))
            ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=7)
            weeks = hm["week_starts"]
            step = max(1, len(weeks) // 12)
            ticks = list(range(0, len(weeks), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([weeks[i].strftime("%Y-%m-%d") for i in ticks],
                               rotation=90, fontsize=7)
            ax.set_title(f"Activity heatmap: {hm['active_days']} active days, "
                         f"busiest day {hm['max_day']} messages", fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8)
            save(fig, "activity_heatmap.png")

    # pace trends
    pace = analyses.get("pace")
    if pace and pace["days"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(range(len(pace["days"])), pace["rolling"], color=PALETTE[0])
            ax.set_title("Average messages per day (30-day rolling)", fontweight="bold")
            step = max(1, len(pace["days"]) // 12)
            ax.set_xticks(range(0, len(pace["days"]), step))
            ax.set_xticklabels([pace["days"][i] for i in range(0, len(pace["days"]), step)],
                               rotation=45, ha="right", fontsize=7)
            save(fig, "pace_trends.png")

    if pace and pace["calls_by_month"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 3))
            months = sorted(pace["calls_by_month"])
            ax.bar(range(len(months)), [pace["calls_by_month"][m] for m in months], color=PALETTE[2])
            ax.set_title("Calls per month", fontweight="bold")
            step = max(1, len(months) // 12)
            ax.set_xticks(range(0, len(months), step))
            ax.set_xticklabels([months[i] for i in range(0, len(months), step)],
                               rotation=45, ha="right", fontsize=7)
            save(fig, "calls_over_time.png")

    if pace and pace["media_by_year"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 3))
            years = sorted(pace["media_by_year"])
            ax.bar(range(len(years)), [pace["media_by_year"][y] for y in years], color=PALETTE[1])
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years)
            ax.set_title("Media sent per year", fontweight="bold")
            save(fig, "media_by_year.png")

    # pair matrices
    pm = analyses.get("pair_matrices")
    if pm:
        def _matrix_plot(matrix, name, title):
            members = pm["members"]
            n = len(members)
            data = np.zeros((n, n))
            for i, a in enumerate(members):
                for j, b in enumerate(members):
                    data[i, j] = matrix[a].get(b, 0)
            with plt.rc_context(theme):
                fig, ax = plt.subplots(figsize=(max(5, n * 0.55), max(4, n * 0.5)))
                im = ax.imshow(data, cmap="YlOrRd")
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(members, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(members, fontsize=8)
                ax.set_xlabel("was replied to / message sender")
                if n <= 15 and data.max() > 0:
                    for i in range(n):
                        for j in range(n):
                            if data[i, j]:
                                ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=7)
                ax.set_title(title, fontweight="bold")
                fig.colorbar(im, ax=ax, shrink=0.8)
                save(fig, name)
        _matrix_plot(pm["reply"], "reply_matrix.png",
                     "Who replies to whom (within 1 hour)")
        _matrix_plot(pm["reaction"], "reaction_matrix.png",
                     "Who reacts to whose messages")

    # hourly radar profiles
    radar = analyses.get("radar")
    if radar:
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False).tolist()
        angles += angles[:1]
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
            for i, (member, hours) in enumerate(sorted(radar.items())):
                vals = hours + hours[:1]
                c = PALETTE[i % len(PALETTE)]
                ax.plot(angles, vals, label=member, color=c)
                ax.fill(angles, vals, alpha=0.08, color=c)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([str(h) for h in range(24)], fontsize=7)
            ax.set_title("Hourly activity profile (hours 0-23)", fontweight="bold")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
            save(fig, "hourly_radar.png")

    # word clouds
    wc_data = analyses.get("wordcloud")
    if wc_data:
        try:
            from wordcloud import WordCloud
        except Exception:
            wc_data = None
        else:
            def save_wc(counter, name, label):
                wc = WordCloud(width=900, height=420, background_color="white",
                               colormap="viridis", max_words=120, random_state=42)
                wc.generate_from_frequencies(dict(counter))
                with plt.rc_context(theme):
                    fig, ax = plt.subplots(figsize=(10, 4.6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title(label, fontweight="bold")
                    save(fig, name)
            save_wc(wc_data["overall"], "wordcloud.png", "Overall word cloud")
            for member, counter in sorted(wc_data["per_member"].items())[:6]:
                save_wc(counter, f"wordcloud_{_slug(member)}.png",
                        f"{member} word cloud")

    # monologues
    mono = analyses.get("monologues")
    if mono and mono["per_member_longest"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_mono = mono["per_member_longest"].most_common(top)
            _bar(fig, ax, [n for n, _ in top_mono], [c for _, c in top_mono],
                 "Longest solo run (messages in a row)")
            save(fig, "monologues.png")

    # unsent
    unsent = analyses.get("unsent")
    if unsent and any(unsent.values()):
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_unsent = unsent.most_common(top)
            _bar(fig, ax, [n for n, _ in top_unsent], [c for _, c in top_unsent],
                 "Unsent messages per member")
            save(fig, "unsent.png")

    # emoji timeline (top 5 emojis across years)
    emo = analyses.get("emojis")
    if emo and emo["per_year"]:
        years = sorted(emo["per_year"])
        top_emoji_names = [e for e, _ in _sum_counters(
            emo["per_year"][y] for y in years).most_common(5)]
        if top_emoji_names:
            with plt.rc_context(theme):
                fig, ax = plt.subplots(figsize=(9, 4))
                for i, e in enumerate(top_emoji_names):
                    series = [emo["per_year"][y].get(e, 0) for y in years]
                    ax.plot(years, series, marker="o", label=emoji_lib.demojize(e).strip(":"),
                            color=PALETTE[i % len(PALETTE)])
                ax.set_xticks(years)
                ax.set_title("Favorite emojis over the years", fontweight="bold")
                ax.legend(fontsize=8)
                save(fig, "emoji_timeline.png")

    # question dynamics
    qst = analyses.get("questions")
    if qst and qst["table"]:
        top_q = qst["table"][:top]
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            _bar(fig, ax, [r["member"] for r in top_q], [r["asked"] for r in top_q],
                 "Questions asked per member")
            save(fig, "questions_asked.png")
        responders = [r for r in qst["table"] if r["median_m"] is not None]
        if responders:
            responders = sorted(responders, key=lambda r: r["median_m"])[:top]
            with plt.rc_context(theme):
                fig, ax = plt.subplots(figsize=(8, 4))
                _bar(fig, ax, [r["member"] for r in responders],
                     [max(0.1, r["median_m"]) for r in responders],
                     "Median time to answer a question (minutes)")
                save(fig, "question_speed.png")

    # topics per year (tf-idf)
    topics = analyses.get("topics")
    if topics and topics["by_year"]:
        show_years = topics["years"][-12:]
        # Pick the columns first, then derive exactly the rows needed. Deriving
        # the columns from a row cap instead leaves a whole row empty.
        n_cols = 1 if len(show_years) == 1 else 2
        n_rows = -(-len(show_years) // n_cols)
        with plt.rc_context(theme):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.4 * n_rows),
                                     constrained_layout=True)
            axes = [axes] if n_rows == n_cols == 1 else list(np.ravel(axes))
            for i, year in enumerate(show_years):
                ax = axes[i]
                words = topics["by_year"][year]
                labels = [w["word"] for w in words]
                counts = [w["count"] for w in words]
                colors = [PALETTE[j % len(PALETTE)] for j in range(len(labels))]
                ax.barh(range(len(labels)), counts[::-1], color=colors[::-1])
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels[::-1], fontsize=8)
                ax.set_title(str(year), fontsize=10, fontweight="bold")
            for j in range(len(show_years), len(axes)):
                axes[j].axis("off")
            fig.suptitle("What the chat was about each year", fontweight="bold")
            save(fig, "topics_by_year.png")

    # running jokes
    jokes = analyses.get("jokes")
    if jokes and jokes["jokes"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_jokes = jokes["jokes"][:8]
            _bar(fig, ax, [j["phrase"] for j in top_jokes],
                 [j["count"] for j in top_jokes], "Running jokes")
            save(fig, "inside_jokes.png")


def _shorten(text, n=40):
    if not text:
        return "(no text)"
    text = text.replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def _media_label(m):
    for flag, name in (("has_photo", "photo"), ("has_sticker", "sticker"),
                       ("has_gif", "GIF"), ("has_video", "video"),
                       ("has_audio", "audio"), ("has_file", "file")):
        if m[flag]:
            return name
    return None


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #

def all_time_totals(stats, analyses):
    """Whole-chat numbers as (label, value) pairs.

    Each of these already existed, but scattered across the section that
    happened to compute it, so nothing answered "how big is this chat" in one
    place. Values come from analyses that --skip can remove, hence the guards.
    """
    react = analyses.get("reactions") or {}
    swear = analyses.get("swear") or {}
    qst = analyses.get("questions") or {}
    conv = analyses.get("conversations") or {}
    emo = analyses.get("emojis") or {}
    active = stats.get("active_days") or 0
    emoji_total = emo.get("total_emojis")
    if emoji_total is None:
        emoji_total = sum(stats["emojis"].values())
    return [
        ("Messages", f"{stats['total']:,}"),
        ("Members", f"{len(stats['member_msgs']):,}"),
        ("Words", f"{stats.get('total_words', 0):,}"),
        ("Emojis", f"{emoji_total:,}"),
        ("Reactions", f"{react.get('total_reactions', 0):,}"),
        ("Questions asked", f"{qst.get('total_questions', 0):,}"),
        ("Swear messages", f"{swear.get('total_hits', 0):,}"),
        ("Media", f"{stats['media']:,}"),
        ("Links shared", f"{stats['links']:,}"),
        ("Calls", f"{stats['calls']:,} ({int(stats['call_seconds'] // 60):,} min)"),
        ("Conversations", f"{conv.get('conversation_count', 0):,}"),
        ("Active days", f"{active:,}"),
        ("Messages per active day",
         f"{stats['total'] / active:,.1f}" if active else "-"),
        ("Longest daily streak", f"{stats['longest_streak']:,} days"),
    ]


def write_summary(title, stats, analyses, track, out_dir, anonymized, dates, top=10):
    lines = [f"# {title} flashback", ""]
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                 + (" (names anonymized)" if anonymized else "") + "")
    lines.append("")
    lines.append("## All-time totals")
    lines.append("")
    lines.append(f"- **Period**: {dates[0]} to {dates[-1]}")
    for label, value in all_time_totals(stats, analyses):
        lines.append(f"- **{label}**: {value}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Member | Messages | Share |")
    lines.append("|---|---|---|")
    total = max(1, stats["total"])
    for member, count in stats["member_msgs"].most_common(top):
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
    for member, count in react["reactor"].most_common(top):
        lines.append(f"| {member} | {count:,} |")
    lines.append("")

    lines.append("## Response speed")
    lines.append("")
    lines.append("Median time to reply, fastest first. Ghosted % is the share of a "
                 "member's turns that got no reply within an hour.")
    lines.append("")
    lines.append("| Member | Replies | Median reply | Replies <5 min | Ghosted % |")
    lines.append("|---|---|---|---|---|")
    for r in analyses["speed"]["table"]:
        med = "-" if r["median_m"] is None else f"{r['median_m']} min"
        fast = "-" if r["fast5_pct"] is None else f"{r['fast5_pct']}%"
        ghost = "-" if r["ghost_pct"] is None else f"{r['ghost_pct']}%"
        lines.append(f"| {r['member']} | {r['replies']} | {med} | {fast} | {ghost} |")
    lines.append("")

    swear = analyses["swear"]
    lines.append("## Swear-word analytics")
    lines.append("")
    if swear["total_hits"]:
        lines.append(f"**{swear['total_hits']}** messages contain profanity.")
        lines.append("")
        lines.append("| Member | Swear messages | Signature swear word |")
        lines.append("|---|---|---|")
        for member, count in swear["member_hits"].most_common(top):
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
            top_user = data["per_member"].most_common(1)[0] if data["per_member"] else ("-", 0)
            lines.append(f"| {term} | {data['count']} | {top_user[0]} ({top_user[1]}) |")
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

    ld = analyses.get("links_domains", {})
    if ld and ld["domains"]:
        lines.append("## Links and domains")
        lines.append("")
        lines.append("| Domain | Shares |")
        lines.append("|---|---|")
        for domain, count in ld["domains"].most_common(top):
            lines.append(f"| {domain} | {count} |")
        lines.append("")

    media = analyses.get("media", {})
    if media and any(media[k] for k in ("photos", "stickers", "gifs", "videos", "audio", "files")):
        lines.append("## Media leaderboard")
        lines.append("")
        lines.append("| Member | Photos | Stickers | GIFs | Videos | Audio | Files |")
        lines.append("|---|---|---|---|---|---|---|")
        for member in sorted(set().union(*[set(media[k]) for k in
                                           ("photos", "stickers", "gifs", "videos", "audio", "files")]),
                             # Name breaks ties so a set's arbitrary iteration
                             # order cannot reshuffle equal rows between runs.
                             key=lambda m: (-sum(media[k].get(m, 0) for k in
                                                 ("photos", "stickers", "gifs",
                                                  "videos", "audio", "files")), m)):
            lines.append(f"| {member} | {media['photos'].get(member, 0)} | "
                         f"{media['stickers'].get(member, 0)} | {media['gifs'].get(member, 0)} | "
                         f"{media['videos'].get(member, 0)} | {media['audio'].get(member, 0)} | "
                         f"{media['files'].get(member, 0)} |")
        lines.append("")

    conv = analyses.get("conversations", {})
    if conv and conv["starters"]:
        lines.append("## Conversation starters")
        lines.append("")
        lines.append(f"A conversation is split on a 30-minute gap. The chat had "
                     f"**{conv['conversation_count']}** separate sessions.")
        lines.append("")
        lines.append("| Member | Sessions started |")
        lines.append("|---|---|")
        for member, count in conv["starters"].most_common(top):
            lines.append(f"| {member} | {count} |")
        if conv["longest_run"]:
            run = conv["longest_run"]
            lines.append("")
            lines.append(f"Longest single session: **{conv['longest_run_len']}** messages "
                         f"({run[0]['dt'].strftime('%Y-%m-%d %H:%M')} - "
                         f"{run[-1]['dt'].strftime('%H:%M')}).")
        lines.append("")

    chains = analyses.get("reply_chains")
    if chains and chains["top_chains"]:
        lines.append("## Reply chains")
        lines.append("")
        lines.append(f"**{chains['count']}** messages were part of a reply chain of 2+.")
        lines.append("")
        for i, chain in enumerate(chains["top_chains"], 1):
            labels = []
            for m in chain:
                if m["content"]:
                    labels.append(f"**{m['sender']}**: {_shorten(m['content'], 60)}")
                else:
                    label = _media_label(m)
                    labels.append(f"**{m['sender']}**: [{label or 'media'}]")
            lines.append(f"{i}. {' -> '.join(labels)}")
            lines.append("")

    ghosts = analyses.get("ghosting", {})
    if ghosts:
        lines.append("## Ghosting stats")
        lines.append("")
        lines.append("Longest silence between a member's own messages, in days:")
        lines.append("")
        lines.append("| Member | Longest silence (days) |")
        lines.append("|---|---|")
        for member, gap in sorted(ghosts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {member} | {gap} |")
        lines.append("")

    lengths = analyses.get("length_trends", {})
    if lengths:
        lines.append("## Message length trends")
        lines.append("")
        lines.append("| Year | Avg chars | Longest |")
        lines.append("|---|---|---|")
        for y in list(lengths):
            lines.append(f"| {y} | {lengths[y]['avg_chars']} | {lengths[y]['max_chars']} |")
        lines.append("")

    senti = analyses.get("sentiment")
    if senti:
        lines.append("## Sentiment (VADER)")
        lines.append("")
        lines.append(f"Scored **{senti['messages_scored']}** messages. English-only; may be noisy "
                     f"on mixed-language chat.")
        lines.append("")
        lines.append("| Member | Avg sentiment (-1 to +1) |")
        lines.append("|---|---|")
        for member, score in sorted(senti["per_member"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {member} | {score:+.3f} |")
        lines.append("")

    ex = analyses.get("extremes")
    if ex:
        lines.append("## Extremes")
        lines.append("")
        if ex["longest"] and ex["longest"]["content"]:
            lines.append(f"- Longest message: **{ex['longest']['sender']}** "
                         f"({ex['longest']['dt'].strftime('%Y-%m-%d')}, "
                         f"{len(ex['longest']['content'])} chars) "
                         f"\"{_shorten(ex['longest']['content'], 80)}\"")
        if ex["most_reacted"]:
            lines.append(f"- Most-reacted: **{ex['most_reacted']['sender']}** "
                         f"({ex['most_reacted']['dt'].strftime('%Y-%m-%d')}, "
                         f"{len(ex['most_reacted']['reactions'])} reactions) "
                         f"\"{_shorten(ex['most_reacted']['content'], 80)}\"")
        lines.append(f"- Record day: **{ex['record_day']}** ({ex['record_day_count']} messages)")
        lines.append("")

    hm = analyses.get("heatmap")
    if hm:
        lines.append("## Activity & pace")
        lines.append("")
        lines.append(f"- **{hm['active_days']}** distinct active days.")
        lines.append(f"- Busiest day: **{hm['max_day']}** messages.")
        pace = analyses.get("pace")
        if pace and pace["days"]:
            avg = sum(pace["counts"]) / len(pace["days"])
            lines.append(f"- Average **{avg:.1f}** messages per active day.")
        if pace and pace["media_by_year"]:
            peak_media = max(pace["media_by_year"].items(), key=lambda kv: kv[1])
            lines.append(f"- Peak media year: **{peak_media[0]}** "
                         f"({peak_media[1]} photos/stickers/GIFs/videos).")
        lines.append("")

    pm = analyses.get("pair_matrices")
    if pm:
        best_reply = max(((a, b, c) for a, row in pm["reply"].items()
                          for b, c in row.items()), key=lambda x: x[2], default=None)
        best_react = max(((a, b, c) for a, row in pm["reaction"].items()
                          for b, c in row.items()), key=lambda x: x[2], default=None)
        lines.append("## Pair dynamics")
        lines.append("")
        lines.append("Row replies to column (within 1 hour), and row reacts to column's "
                     "messages:")
        lines.append("")
        if best_reply:
            lines.append(f"- Most replies: **{best_reply[0]}** -> **{best_reply[1]}** "
                         f"({best_reply[2]} times).")
        if best_react:
            lines.append(f"- Most reactions: **{best_react[0]}** -> **{best_react[1]}** "
                         f"({best_react[2]} times).")
        lines.append("")

    radar = analyses.get("radar")
    if radar:
        peak_hours = []
        for member, hours in radar.items():
            peak = max(range(24), key=lambda h: hours[h])
            peak_hours.append((member, peak))
        lines.append("## Hourly profiles")
        lines.append("")
        lines.append("| Member | Peak hour |")
        lines.append("|---|---|")
        for member, hour in sorted(peak_hours):
            lines.append(f"| {member} | {hour}:00 |")
        lines.append("")

    mono = analyses.get("monologues")
    if mono and mono["longest_run"]:
        lines.append("## Monologues")
        lines.append("")
        lines.append(f"**{mono['email_moments']}** moment(s) where someone went 4+ "
                     f"messages solo ('could've been an email').")
        lines.append("")
        lines.append("| Member | Longest solo run |")
        lines.append("|---|---|")
        for member, length in mono["per_member_longest"].most_common(top):
            lines.append(f"| {member} | {length} |")
        run = mono["longest_run"]
        lines.append("")
        lines.append(f"Record solo run: **{run[0]['sender']}** with {len(run)} messages "
                     f"({run[0]['dt'].strftime('%Y-%m-%d %H:%M')}).")
        lines.append("")

    unsent = analyses.get("unsent")
    if unsent and any(unsent.values()):
        lines.append("## Unsent messages")
        lines.append("")
        lines.append("Messages typed but unsent (`is_unsent`):")
        lines.append("")
        lines.append("| Member | Unsent |")
        lines.append("|---|---|")
        for member, count in unsent.most_common(top):
            lines.append(f"| {member} | {count} |")
        lines.append("")

    taken_down = analyses.get("taken_down")
    if taken_down and any(taken_down.values()):
        lines.append("## Removed messages")
        lines.append("")
        lines.append("Messages flagged `is_taken_down` (no longer visible in the chat):")
        lines.append("")
        lines.append("| Member | Removed |")
        lines.append("|---|---|")
        for member, count in taken_down.most_common(top):
            lines.append(f"| {member} | {count} |")
        lines.append("")

    emo = analyses.get("emojis")
    if emo and emo["per_member"]:
        lines.append("## Emoji report")
        lines.append("")
        lines.append(f"**{emo['total_emojis']:,}** emojis in total. Emojis per 100 messages:")
        lines.append("")
        lines.append("| Member | Emojis | Emojis/100 msgs | Top 3 emojis |")
        lines.append("|---|---|---|---|")
        for member, counter in sorted(emo["per_member"].items(),
                                       key=lambda kv: -sum(kv[1].values())):
            top3 = ", ".join(e for e, _ in counter.most_common(3))
            per100 = emo["emojis_per_100"].get(member, 0)
            lines.append(f"| {member} | {sum(counter.values()):,} | {per100} | {top3} |")
        lines.append("")

    qst = analyses.get("questions")
    if qst and qst["table"]:
        lines.append("## Question dynamics")
        lines.append("")
        lines.append(f"**{qst['total_questions']}** questions asked; "
                     f"**{qst['total_answered']}** got a reply within an hour "
                     f"({100 * qst['total_answered'] / max(1, qst['total_questions']):.0f}%).")
        lines.append("")
        lines.append("| Member | Asked | Answered | Answer % | Responses given | Median answer |")
        lines.append("|---|---|---|---|---|---|")
        for r in qst["table"]:
            ans = "n/a" if r["answer_pct"] is None else f"{r['answer_pct']}%"
            med = "n/a" if r["median_m"] is None else f"{r['median_m']} min"
            lines.append(f"| {r['member']} | {r['asked']} | {r['answered']} | {ans} "
                         f"| {r['responses_given']} | {med} |")
        lines.append("")

    topics = analyses.get("topics")
    if topics and topics["by_year"]:
        lines.append("## What the chat was about")
        lines.append("")
        lines.append("Top topic words per year (tf-idf):")
        lines.append("")
        for year, words in topics["by_year"].items():
            bits = ", ".join(f"**{w['word']}** ({w['count']})" for w in words)
            lines.append(f"- **{year}**: {bits}")
        lines.append("")

    jokes = analyses.get("jokes")
    if jokes and jokes["jokes"]:
        lines.append("## Running jokes")
        lines.append("")
        lines.append("Phrases said often enough, by enough people, over enough years "
                     "to count as an inside joke:")
        lines.append("")
        lines.append("| Phrase | Times | Members | Years |")
        lines.append("|---|---|---|---|")
        for j in jokes["jokes"]:
            lines.append(f"| *{j['phrase']}* | {j['count']} | "
                         f"{', '.join(j['members'])} | {', '.join(str(y) for y in j['years'])} |")
        example = jokes["jokes"][0].get("example")
        if example:
            lines.append("")
            lines.append(f"Example: \"{example}\"")
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    charts = sorted(p.name for p in out_dir.glob("*.png"))
    for c in charts:
        lines.append(f"![{c}]({c})")
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _base64_png(path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_YEAR_PAGE_CSS = """
*{box-sizing:border-box}
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:0 0 60px;color:#202124;background:#fff;line-height:1.5}
[data-theme="dark"]{color:#e8e8e8;background:#17191f}
.topbar{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:16px;padding:8px 20px;background:#f7f8fa;border-bottom:1px solid #e3e5e8;flex-wrap:wrap}
[data-theme="dark"] .topbar{background:#20242d;border-bottom:1px solid #2a2f3a}
.brand{font-weight:700;font-size:15px}
.topbar a{color:#5b8ff9;text-decoration:none;font-size:12px}
main{max-width:860px;margin:0 auto;padding:0 20px}
h1{font-size:26px;margin-bottom:4px}
h2{font-size:20px;margin-top:36px;border-bottom:1px solid #e3e5e8;padding-bottom:6px}
.muted{color:#777;font-size:13px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.card{flex:1;min-width:120px;background:#f7f8fa;border:1px solid #e3e5e8;border-radius:12px;padding:14px}
.card b{display:block;font-size:22px}
.card span{font-size:12px;color:#777}
[data-theme="dark"] .card{background:#20242d;border-color:#2a2f3a}
[data-theme="dark"] .card span{color:#9aa0a6}
[data-theme="dark"] h2{border-color:#2a2f3a}
ul{margin:8px 0}
li{margin:4px 0}
img{max-width:100%;border-radius:10px;margin:6px 0}
table{border-collapse:collapse;width:100%;margin:10px 0}
th,td{border:1px solid #e3e5e8;padding:6px 10px;text-align:left;font-size:13px}
[data-theme="dark"] th,[data-theme="dark"] td{border-color:#2a2f3a}
#theme{border:1px solid #e3e5e8;background:#fff;color:#202124;border-radius:6px;padding:4px 10px;cursor:pointer;margin-left:auto}
"""


def write_year_reviews(title, stats, analyses, out_dir):
    years = sorted(stats["by_year"])
    if not years:
        return
    index_rows = []
    for year in years:
        recap = analyses.get("yearly", {}).get(year, {})
        pngs = _year_mini_charts(stats, analyses, year)
        html_doc = _year_page_html(title, year, recap, analyses, pngs)
        (out_dir / f"year_{year}.html").write_text(html_doc, encoding="utf-8")
        top = recap.get("top_member", "-")
        index_rows.append(
            f"<tr><td><a href='year_{year}.html'>{year}</a></td>"
            f"<td>{recap.get('total', 0):,}</td>"
            f"<td>{html_lib.escape(str(top))}</td>"
            f"<td>{recap.get('active_members', 0)}</td>"
            f"<td>{recap.get('record_day', '-')}</td></tr>")
    rows = "".join(index_rows)
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)} - year in review</title>
<style>{_YEAR_PAGE_CSS}</style>
</head><body>
<div class="topbar"><span class="brand">{html_lib.escape(title)} flashback</span>
<a href="report.html">Report</a>
<button id="theme" title="Toggle theme" aria-label="Toggle theme">Dark</button></div>
<main>
<h1>Year in review</h1>
<p class="muted">One page per year of chat history.</p>
<table>
<tr><th>Year</th><th>Messages</th><th>Top member</th><th>Active members</th><th>Record day</th></tr>
{rows}
</table>
</main>
<script>
var t=document.getElementById('theme');
t.onclick=function(){{var dark=document.body.dataset.theme!=='dark';document.body.dataset.theme=dark?'dark':'light';t.textContent=dark?'Light':'Dark';}};
</script>
</body></html>"""
    (out_dir / "year_in_review.html").write_text(index, encoding="utf-8")


def _year_mini_charts(stats, analyses, year):
    pngs = []
    months = list(range(1, 13))
    counts = [0] * 12
    for d, c in stats["by_day"].items():
        if d.year == year:
            counts[d.month - 1] += c
    if any(counts):
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        ax.bar(months, counts, color=PALETTE[0])
        ax.set_xticks(months)
        ax.set_title(f"Messages per month in {year}", fontweight="bold")
        pngs.append(("Monthly activity", _fig_to_png(fig)))

    topics = (analyses.get("topics") or {}).get("by_year", {}).get(year)
    if topics:
        words = [t["word"] for t in topics[:8]]
        vals = [t["count"] for t in topics[:8]]
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        _bar(fig, ax, words, vals, "Top words of the year")
        pngs.append(("Top words", _fig_to_png(fig)))

    emojis = (analyses.get("emojis") or {}).get("per_year", {}).get(year)
    if emojis:
        top = emojis.most_common(8)
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        _bar(fig, ax, [emoji_lib.demojize(e).strip(":") for e, _ in top],
             [c for _, c in top], "Top emojis of the year")
        pngs.append(("Top emojis", _fig_to_png(fig)))
    return pngs


def _year_page_html(title, year, recap, analyses, pngs):
    cards = []
    cards.append(f"<div class='card'><b>{recap.get('total', 0):,}</b><span>messages</span></div>")
    cards.append(f"<div class='card'><b>{recap.get('active_members', 0)}</b><span>members active</span></div>")
    cards.append(f"<div class='card'><b>{html_lib.escape(str(recap.get('top_member', '-')))}</b><span>top member</span></div>")
    cards.append(f"<div class='card'><b>{recap.get('record_day', '-')}</b><span>record day</span></div>")

    facts = []
    top_word = recap.get("top_word")
    if top_word:
        facts.append(f"<li>Top word: <b>{html_lib.escape(top_word[0])}</b> "
                     f"({top_word[1]}x)</li>")
    top_emoji = recap.get("top_emoji")
    if top_emoji:
        facts.append(f"<li>Top emoji: <b>{top_emoji[0]}</b> ({top_emoji[1]}x)</li>")
    record_count = recap.get("record_day_count")
    if record_count:
        facts.append(f"<li>Busiest day: <b>{recap.get('record_day')}</b> "
                     f"with {record_count} messages</li>")
    reacted = recap.get("best_reacted")
    if reacted:
        facts.append(f"<li>Most-reacted: <b>{html_lib.escape(reacted['sender'])}</b> "
                     f"({len(reacted['reactions'])} reactions) "
                     f"&quot;{html_lib.escape(_shorten(reacted.get('content'), 70))}&quot;</li>")

    chart_html = "".join(
        f"<figure><img src='data:image/png;base64,{b64}' alt='{html_lib.escape(label)}'/>"
        f"<figcaption class='muted'>{html_lib.escape(label)}</figcaption></figure>"
        for label, b64 in pngs)

    jokes = (analyses.get("jokes") or {}).get("jokes", [])
    year_jokes = [j for j in jokes if year in j["years"]]
    jokes_html = ""
    if year_jokes:
        rows = "".join(
            f"<tr><td><i>{html_lib.escape(j['phrase'])}</i></td><td>{j['count']}</td>"
            f"<td>{html_lib.escape(', '.join(j['members']))}</td></tr>"
            for j in year_jokes)
        jokes_html = f"<h2>Running jokes</h2><table><tr><th>Phrase</th><th>Times</th><th>Members</th></tr>{rows}</table>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)} - {year} in review</title>
<style>{_YEAR_PAGE_CSS}</style>
</head><body>
<div class="topbar"><span class="brand">{html_lib.escape(title)} flashback</span>
<a href="year_in_review.html">All years</a>
<a href="report.html">Report</a>
<button id="theme" title="Toggle theme" aria-label="Toggle theme">Dark</button></div>
<main>
<h1>{year} in review</h1>
<p class="muted">{html_lib.escape(title)}</p>
<div class="cards">{''.join(cards)}</div>
{"<ul>" + "".join(facts) + "</ul>" if facts else ""}
{chart_html}
{jokes_html}
</main>
<script>
var t=document.getElementById('theme');
t.onclick=function(){{var dark=document.body.dataset.theme!=='dark';document.body.dataset.theme=dark?'dark':'light';t.textContent=dark?'Light':'Dark';}};
</script>
</body></html>"""


def _js_msg(m):
    if m is None:
        return None
    return {
        "sender": m["sender"],
        "ts": m["dt"].strftime("%Y-%m-%d %H:%M"),
        "content": m["content"],
        "mtype": m["mtype"],
        "reactions": len(m["reactions"]),
    }


def insights(stats, analyses):
    out = []
    if stats["member_msgs"]:
        top = stats["member_msgs"].most_common(1)[0]
        out.append(f"{top[0]} leads the chat with {top[1]:,} messages "
                   f"({100 * top[1] / max(1, stats['total']):.0f}% of the chat).")
    conv = analyses.get("conversations", {})
    if conv and conv["longest_run_len"]:
        out.append(f"The longest single session ran {conv['longest_run_len']} messages.")
    fastest = _fastest_replier(analyses.get("speed"))
    if fastest:
        out.append(f"{fastest['member']} answers fastest with a "
                   f"{fastest['median_m']} min median reply time.")
    ghosts = analyses.get("ghosting", {})
    if ghosts:
        worst = max(ghosts.items(), key=lambda kv: kv[1])
        out.append(f"{worst[0]} holds the longest silent streak ({worst[1]} days).")
    mono = analyses.get("monologues", {})
    if mono and mono["per_member_longest"]:
        top_mono = mono["per_member_longest"].most_common(1)[0]
        out.append(f"{top_mono[0]} once went on a {top_mono[1]}-message solo run.")
    pm = analyses.get("pair_matrices")
    if pm:
        best_pair = max(((a, b, c) for a, row in pm["reply"].items()
                         for b, c in row.items()), key=lambda x: x[2], default=None)
        if best_pair:
            out.append(f"{best_pair[0]} replies to {best_pair[1]} more than anyone "
                       f"({best_pair[2]} times).")
    senti = analyses.get("sentiment")
    if senti and senti["per_member"]:
        best = max(senti["per_member"].items(), key=lambda kv: kv[1])
        out.append(f"Best vibes come from {best[0]} ({best[1]:+.2f} avg sentiment).")
    ex = analyses.get("extremes")
    if ex:
        out.append(f"The record day was {ex['record_day']} with "
                   f"{ex['record_day_count']} messages.")
    emo = analyses.get("emojis")
    if emo and emo["per_year"]:
        top_emoji = _sum_counters(emo["per_year"].values()).most_common(1)
        if top_emoji:
            out.append(f"The chat's favorite emoji is {top_emoji[0][0]} "
                       f"({top_emoji[0][1]} uses).")
    qst = analyses.get("questions")
    if qst and qst["table"]:
        by_rate = [r for r in qst["table"] if r["asked"] >= 5 and r["answer_pct"] is not None]
        if by_rate:
            worst = min(by_rate, key=lambda r: r["answer_pct"])
            out.append(f"{worst['member']} gets left on read the most "
                       f"({100 - worst['answer_pct']:.0f}% of questions unanswered).")
    jokes = analyses.get("jokes")
    if jokes and jokes["jokes"]:
        top_joke = jokes["jokes"][0]
        out.append(f"Running joke: \"{top_joke['phrase']}\" "
                   f"({top_joke['count']} times by {len(top_joke['members'])} people).")
    return out


CHART_CAPTIONS = {
    "messages_by_year.png": "Messages per year",
    "activity_by_hour.png": "When the chat is alive, by hour",
    "activity_by_weekday.png": "Busiest days of the week",
    "activity_heatmap.png": "Day-by-day activity calendar",
    "pace_trends.png": "Rolling average messages per day",
    "monthly_timeline.png": "Messages per month over the years",
    "top_members.png": "Who talks the most",
    "top_words.png": "Favorite words",
    "top_emojis.png": "Favorite emojis",
    "wordcloud.png": "Overall word cloud",
    "yearly_recap.png": "Year-by-year recap",
    "reactions_given.png": "Who reacts the most",
    "most_reacted.png": "The most-reacted messages",
    "reaction_matrix.png": "Who reacts to whose messages",
    "response_speed.png": "Median time to reply",
    "reply_matrix.png": "Who replies to whom",
    "reply_chains.png": "Longest reply chains",
    "hourly_radar.png": "Hourly activity profiles",
    "conversation_starters.png": "Who starts conversations",
    "ghosting.png": "Longest silent streaks",
    "monologues.png": "Longest solo runs",
    "swear_by_member.png": "Swear words per member",
    "swear_over_time.png": "Swearing over the years",
    "tracked_terms.png": "Custom tracked terms per year",
    "sentiment_per_member.png": "Average mood per member",
    "sentiment_over_time.png": "Mood over the years",
    "length_trends.png": "Average message length",
    "word_trends.png": "Top words over time",
    "top_domains.png": "Most shared domains",
    "media_leaderboard.png": "Media sent per member",
    "media_by_year.png": "Media per year",
    "calls_over_time.png": "Calls per month",
    "unsent.png": "Unsent messages per member",
    "emoji_timeline.png": "Favorite emojis over the years",
    "questions_asked.png": "Questions asked per member",
    "question_speed.png": "Median time to answer a question",
    "topics_by_year.png": "What the chat was about each year",
    "inside_jokes.png": "Running jokes",
}


def _sec(sid, title, inner):
    return f'<section id="{sid}"><h2>{title}</h2>{inner}</section>'


def _thead(columns):
    return "<thead><tr>" + "".join(f"<th>{html_lib.escape(c)}</th>" for c in columns) + "</tr></thead>"


def _table(columns, rows):
    body = "<tbody>" + "".join("<tr>" + "".join(f"<td>{r}</td>" for r in row) + "</tr>"
                               for row in rows) + "</tbody>"
    return f"<table>{_thead(columns)}{body}</table>"


def write_report_html(title, stats, analyses, out_dir, anonymized, dates, top=10):
    charts = sorted(out_dir.glob("*.png"))
    imgs = "".join(
        f'<figure><img loading="lazy" src="data:image/png;base64,{_base64_png(c)}" '
        f'alt="{html_lib.escape(c.name)}"/>'
        f"<figcaption>{html_lib.escape(CHART_CAPTIONS.get(c.name, c.name))}</figcaption></figure>"
        for c in charts
    )
    react = analyses["reactions"]
    senti = analyses.get("sentiment")
    ex = analyses.get("extremes")

    leader_rows = [(html_lib.escape(m), f"{c:,}", f"{100 * c / max(1, stats['total']):.1f}%")
                   for m, c in stats["member_msgs"].most_common(top)]
    reactor_rows = [(html_lib.escape(m), f"{c:,}") for m, c in react["reactor"].most_common(top)]

    sections = []
    sections.append(_sec("highlights", "Highlights",
                         "<ul>" + "".join(f"<li>{html_lib.escape(i)}</li>"
                                          for i in insights(stats, analyses)) + "</ul>"))
    sections.append(_sec("leaderboard", "Leaderboard",
                         _table(["Member", "Messages", "Share"], leader_rows)))
    sections.append(_sec("reactive", "Most reactive",
                         _table(["Reactor", "Reactions"], reactor_rows)))

    pm = analyses.get("pair_matrices")
    if pm:
        rows = []
        for a in pm["members"]:
            row = [html_lib.escape(a)]
            for b in pm["members"]:
                n = pm["reply"][a].get(b, 0)
                row.append(str(n))
            rows.append(row)
        sections.append(_sec("pair_dynamics", "Pair dynamics",
                             _table(["Replier \\n Replied-to"] + [html_lib.escape(m) for m in pm["members"]], rows)))
    mono = analyses.get("monologues")
    if mono and mono["per_member_longest"]:
        rows = [(html_lib.escape(m), str(n))
                for m, n in mono["per_member_longest"].most_common(top)]
        sections.append(_sec("monologues", "Monologues",
                             _table(["Member", "Longest solo run"], rows)))

    weird_items = "".join(
        f"<li><b>{html_lib.escape(s['member'])}</b> ({s['dt'].strftime('%Y-%m-%d %H:%M')}) "
        f"[{html_lib.escape(', '.join(s['reasons']))}] "
        f'"{html_lib.escape(s["snippet"])}"</li>'
        for s in analyses["weird"]
    )
    if weird_items:
        sections.append(_sec("weirdest", "Weirdest statements", f"<ul>{weird_items}</ul>"))

    extremes_block = ""
    if ex:
        bits = []
        if ex["longest"] and ex["longest"]["content"]:
            bits.append(f"Longest message: <b>{html_lib.escape(ex['longest']['sender'])}</b> "
                        f"({len(ex['longest']['content'])} chars) "
                        f'"{html_lib.escape(_shorten(ex["longest"]["content"], 80))}"')
        if ex["most_reacted"]:
            bits.append(f"Most-reacted: <b>{html_lib.escape(ex['most_reacted']['sender'])}</b> "
                        f"({len(ex['most_reacted']['reactions'])} reactions) "
                        f'"{html_lib.escape(_shorten(ex["most_reacted"]["content"], 80))}"')
        bits.append(f"Record day: <b>{ex['record_day']}</b> "
                    f"({ex['record_day_count']} messages)")
        extremes_block = "<p>" + "</p><p>".join(bits) + "</p>"
        sections.append(_sec("extremes", "Extremes", extremes_block))

    if senti:
        rows = [(html_lib.escape(m), f"{score:+.3f}")
                for m, score in sorted(senti["per_member"].items(), key=lambda kv: -kv[1])]
        sections.append(_sec("sentiment", "Sentiment",
                             f"<p class='muted'>Scored {senti['messages_scored']} messages; "
                             f"English-only VADER, may be noisy on mixed-language chat.</p>"
                             + _table(["Member", "Avg sentiment"], rows)))
    emo = analyses.get("emojis")
    if emo and emo["per_member"]:
        rows = []
        for member, counter in sorted(emo["per_member"].items(),
                                      key=lambda kv: -sum(kv[1].values())):
            rows.append((html_lib.escape(member), f"{sum(counter.values()):,}",
                         str(emo["emojis_per_100"].get(member, 0)),
                         " ".join(e for e, _ in counter.most_common(3))))
        sections.append(_sec("emojis", "Emoji report",
                             _table(["Member", "Emojis", "Emojis/100", "Top emojis"], rows)))
    qst = analyses.get("questions")
    if qst and qst["table"]:
        rows = []
        for r in qst["table"]:
            ans = "n/a" if r["answer_pct"] is None else f"{r['answer_pct']}%"
            med = "n/a" if r["median_m"] is None else f"{r['median_m']} min"
            rows.append((html_lib.escape(r["member"]), str(r["asked"]), str(r["answered"]),
                         ans, str(r["responses_given"]), med))
        sections.append(_sec("questions", "Question dynamics",
                             f"<p class='muted'>{qst['total_questions']} questions asked; "
                             f"{qst['total_answered']} answered within an hour.</p>"
                             + _table(["Member", "Asked", "Answered", "Answer %",
                                       "Responses", "Median answer"], rows)))
    topics = analyses.get("topics")
    if topics and topics["by_year"]:
        items = "".join(
            f"<li><b>{y}</b>: " + ", ".join(
                f"{html_lib.escape(w['word'])} ({w['count']})" for w in words) + "</li>"
            for y, words in topics["by_year"].items())
        sections.append(_sec("topics", "What the chat was about",
                             "<ul>" + items + "</ul>"))
    jokes = analyses.get("jokes")
    if jokes and jokes["jokes"]:
        rows = []
        for j in jokes["jokes"]:
            rows.append((f"<i>{html_lib.escape(j['phrase'])}</i>", str(j["count"]),
                         html_lib.escape(", ".join(j["members"])),
                         html_lib.escape(", ".join(str(y) for y in j["years"]))))
        example = jokes["jokes"][0].get("example")
        ex_html = (f"<p class='muted'>Example: &quot;{html_lib.escape(example)}&quot;</p>"
                   if example else "")
        sections.append(_sec("jokes", "Running jokes",
                             ex_html + _table(["Phrase", "Times", "Members", "Years"], rows)))
    media = analyses.get("media", {})
    if media and any(media[k] for k in ("photos", "stickers", "gifs", "videos", "audio", "files")):
        rows = []
        for member in sorted(set().union(*[set(media[k]) for k in
                                           ("photos", "stickers", "gifs", "videos", "audio", "files")]),
                             key=lambda m: (-sum(media[k].get(m, 0) for k in
                                                 ("photos", "stickers", "gifs",
                                                  "videos", "audio", "files")), m)):
            rows.append((html_lib.escape(member),
                         *[str(media[k].get(member, 0)) for k in
                           ("photos", "stickers", "gifs", "videos", "audio", "files")]))
        sections.append(_sec("media", "Media leaderboard",
                             _table(["Member", "Photos", "Stickers", "GIFs",
                                     "Videos", "Audio", "Files"], rows)))
    sections.append(_sec("charts", "Charts", imgs))

    nav = "".join(
        f'<a href="#{sid}">{html_lib.escape(label)}</a>'
        for sid, label in [("highlights", "Highlights"), ("leaderboard", "Leaderboard"),
                           ("reactive", "Reactive"), ("pair_dynamics", "Pairs"),
                           ("monologues", "Monologues"), ("weirdest", "Weirdest"),
                           ("extremes", "Extremes"), ("sentiment", "Sentiment"),
                           ("emojis", "Emoji report"), ("questions", "Questions"),
                           ("topics", "Topics"), ("jokes", "Jokes"),
                           ("media", "Media"), ("charts", "Charts")]
    )
    totals_cards = "".join(
        f"<div class='card'><b>{html_lib.escape(value)}</b>"
        f"<span>{html_lib.escape(label.lower())}</span></div>"
        for label, value in all_time_totals(stats, analyses))
    body = f"""<div class="topbar"><span class="brand">{html_lib.escape(title)} flashback</span>
<nav>{nav}</nav>
<a class="years" href="year_in_review.html">Years</a>
<button id="theme" title="Toggle theme" aria-label="Toggle theme">Dark</button></div>
<main>
<h1>{html_lib.escape(title)} flashback</h1>
<p class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
{"(names anonymized)" if anonymized else ""}</p>
<div class="cards">
<div class="card"><b>{dates[0]}</b><span>start</span></div>
<div class="card"><b>{dates[-1]}</b><span>end</span></div>
{totals_cards}
</div>
{''.join(sections)}
</main>"""
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)} flashback</title>
<style>
:root{{--bg:#ffffff;--fg:#202124;--muted:#777;--card:#f7f8fa;--border:#e3e5e8;--accent:#5b8ff9}}
[data-theme="dark"]{{--bg:#17191f;--fg:#e8e8e8;--muted:#9aa0a6;--card:#20242d;--border:#2a2f3a}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:0 0 60px;color:var(--fg);background:var(--bg);line-height:1.5}}
.topbar{{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:16px;padding:8px 20px;background:var(--card);border-bottom:1px solid var(--border);flex-wrap:wrap}}
.brand{{font-weight:700;font-size:15px}}
nav{{display:flex;gap:4px;flex-wrap:wrap;flex:1}}
nav a{{color:var(--muted);text-decoration:none;font-size:12px;padding:4px 8px;border-radius:6px}}
nav a:hover{{background:var(--border);color:var(--fg)}}
.years{{margin-left:auto;color:var(--muted);text-decoration:none;font-size:12px;padding:4px 8px;border:1px solid var(--border);border-radius:6px}}
#theme{{border:1px solid var(--border);background:var(--bg);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}}
main{{max-width:1040px;margin:0 auto;padding:0 20px}}
h1{{font-size:26px;margin-bottom:4px}}
h2{{font-size:20px;margin-top:36px;border-bottom:1px solid var(--border);padding-bottom:6px}}
.muted{{color:var(--muted);font-size:13px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 16px;min-width:110px}}
.card b{{display:block;font-size:22px}} .card span{{color:var(--muted);font-size:12px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border)}}
th{{color:var(--muted);cursor:pointer;user-select:none;white-space:nowrap}}
th.sort-asc::after{{content:" \\25b2";font-size:9px}} th.sort-desc::after{{content:" \\25bc";font-size:9px}}
.tfilter{{margin:8px 0 2px;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--fg);width:220px}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:22px}}
figure{{margin:0;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px}}
figure img{{max-width:100%;border-radius:6px;display:block}}
figcaption{{color:var(--muted);font-size:12px;margin-top:8px}}
li{{margin:6px 0;font-size:14px}}
@media(prefers-color-scheme:dark){{:root:not([data-theme]){{--bg:#17191f;--fg:#e8e8e8;--muted:#9aa0a6;--card:#20242d;--border:#2a2f3a}}}}
</style></head><body>{body}
<script>
(function(){{var r=document.documentElement,b=document.getElementById('theme');
function apply(t){{if(t==='dark'){{r.setAttribute('data-theme','dark');b.textContent='Light';}}else{{r.removeAttribute('data-theme');b.textContent='Dark';}}}}
if(localStorage.getItem('cf-theme'))apply(localStorage.getItem('cf-theme'));
b.addEventListener('click',function(){{var t=r.getAttribute('data-theme')==='dark'?'light':'dark';apply(t);localStorage.setItem('cf-theme',t);}});
document.querySelectorAll('table').forEach(function(t){{var tb=t.tBodies[0];
var box=document.createElement('input');box.placeholder='Filter rows';box.className='tfilter';
t.parentNode.insertBefore(box,t);
box.addEventListener('input',function(){{var q=box.value.toLowerCase();
Array.prototype.forEach.call(tb.rows,function(tr){{tr.style.display=tr.innerText.toLowerCase().indexOf(q)>=0?'':'none';}});}});
var ths=t.querySelectorAll('th');
ths.forEach(function(th,i){{th.addEventListener('click',function(){{var rows=Array.prototype.slice.call(tb.rows);
var num=rows.every(function(tr){{return tr.cells[i]&&!isNaN(parseFloat(tr.cells[i].textContent.replace(/[^0-9.-]/g,'')))&&tr.cells[i].textContent.trim()!=='';}});
var dir=th.getAttribute('data-dir')==='asc'?-1:1;th.setAttribute('data-dir',dir===1?'asc':'desc');
ths.forEach(function(x){{x.classList.remove('sort-asc','sort-desc');}});th.classList.add(dir===1?'sort-asc':'sort-desc');
rows.sort(function(a,b){{var av=a.cells[i].textContent.trim(),bv=b.cells[i].textContent.trim();
return num?(parseFloat(av)-parseFloat(bv))*dir:av.localeCompare(bv)*dir;}});
rows.forEach(function(tr){{tb.appendChild(tr);}});}});}});}});
}})();
</script></body></html>"""
    (out_dir / "report.html").write_text(doc, encoding="utf-8")


def write_summary_json(title, stats, analyses, out_dir, anonymized, dates, top=10):
    react = analyses["reactions"]
    senti = analyses.get("sentiment")
    ex = analyses.get("extremes")
    chains = analyses.get("reply_chains")
    payload = {
        "title": title,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "anonymized": anonymized,
        "period": {"start": dates[0], "end": dates[-1]},
        "total_messages": stats["total"],
        "totals": {label: value for label, value in all_time_totals(stats, analyses)},
        "total_words": stats.get("total_words", 0),
        "active_days": stats.get("active_days", 0),
        "members": len(stats["member_msgs"]),
        "longest_streak_days": stats["longest_streak"],
        "media": stats["media"],
        "calls": stats["calls"],
        "leaderboard": [{"member": m, "messages": c}
                        for m, c in stats["member_msgs"].most_common(top)],
        "yearly": {y: {
            "total": r["total"], "active_members": r["active_members"],
            "top_member": r["top_member"],
            "top_word": {"word": r["top_word"][0], "count": r["top_word"][1]} if r["top_word"] else None,
            "top_emoji": {"emoji": r["top_emoji"][0], "count": r["top_emoji"][1]} if r["top_emoji"] else None,
            "record_day": str(r["record_day"]) if r["record_day"] else None,
            "record_day_count": r["record_day_count"],
            "best_reacted": _js_msg(r["best_reacted"]),
        } for y, r in analyses["yearly"].items()},
        "personality": {
            m: {"messages": p["total_msgs"], "avg_words": p["avg_words"],
                "peak_hour": p["peak_hour"], "night_pct": p["night_pct"],
                "signature_word": p["signature"], "top_emojis": p["top_emojis"]}
            for m, p in analyses["personality"].items()
        },
        "reactions": {
            "total": react["total_reactions"], "reacted_messages": react["total_reacted"],
            "most_reactive": react["most_reactive"],
            "reactors": [{"member": m, "count": c} for m, c in react["reactor"].most_common(top)],
            "emoji_breakdown": [{"emoji": e, "count": c} for e, c in react["reaction_emoji"].most_common(20)],
        },
        "response_speed": [
            {"member": r["member"], "replies": r["replies"], "median_seconds": r["median_s"],
             "fast5_pct": r["fast5_pct"], "ghost_pct": r["ghost_pct"]}
            for r in analyses["speed"]["table"]
        ],
        "swear_words": {
            "total_hits": analyses["swear"]["total_hits"],
            "per_member": [{"member": m, "count": c}
                           for m, c in analyses["swear"]["member_hits"].most_common(top)],
        },
        "weirdest_statements": [
            {"member": s["member"], "ts": s["dt"].strftime("%Y-%m-%d %H:%M"),
             "score": s["score"], "reasons": s["reasons"], "snippet": s["snippet"]}
            for s in analyses["weird"]
        ],
        "links": {
            "top_domains": [{"domain": d, "count": c}
                            for d, c in analyses.get("links_domains", {}).get("domains", {}).most_common(top)],
            "top_links": [{"url": u, "count": c}
                          for u, c in analyses.get("links_domains", {}).get("links", {}).most_common(top)],
        },
        "media": {m: {"photos": analyses["media"]["photos"].get(m, 0),
                      "stickers": analyses["media"]["stickers"].get(m, 0),
                      "gifs": analyses["media"]["gifs"].get(m, 0),
                      "videos": analyses["media"]["videos"].get(m, 0),
                      "audio": analyses["media"]["audio"].get(m, 0),
                      "files": analyses["media"]["files"].get(m, 0)}
                  for m in sorted(set().union(*[set(analyses["media"][k]) for k in
                                                ("photos", "stickers", "gifs", "videos", "audio", "files")]))},
        "conversations": {
            "count": analyses["conversations"]["conversation_count"],
            "longest_run_msgs": analyses["conversations"]["longest_run_len"],
            "starters": [{"member": m, "count": c}
                         for m, c in analyses["conversations"]["starters"].most_common(top)],
        },
        "reply_chains": ({"count": chains["count"], "longest": [len(c) for c in chains["top_chains"]]}
                         if chains else None),
        "ghosting_days": analyses["ghosting"],
        "length_trends": analyses["length_trends"],
        "sentiment": ({"scored": senti["messages_scored"],
                       "per_member": senti["per_member"], "per_year": senti["per_year"]}
                      if senti else None),
        "extremes": ({"longest": _js_msg(ex["longest"]), "shortest": _js_msg(ex["shortest"]),
                      "most_reacted": _js_msg(ex["most_reacted"]),
                      "record_day": str(ex["record_day"]), "record_day_count": ex["record_day_count"]}
                     if ex else None),
        "heatmap": ({"active_days": analyses["heatmap"]["active_days"],
                     "max_day": analyses["heatmap"]["max_day"]}
                    if analyses.get("heatmap") else None),
        "pace": ({"avg_per_active_day": round(sum(analyses["pace"]["counts"]) /
                                             len(analyses["pace"]["days"]), 2),
                  "calls_by_month": dict(analyses["pace"]["calls_by_month"]),
                  "media_by_year": dict(analyses["pace"]["media_by_year"])}
                 if analyses.get("pace") and analyses["pace"]["days"] else None),
        "pair_matrices": ({"members": analyses["pair_matrices"]["members"],
                           "reply": {a: dict(row) for a, row in analyses["pair_matrices"]["reply"].items()},
                           "reaction": {a: dict(row) for a, row in analyses["pair_matrices"]["reaction"].items()}}
                          if analyses.get("pair_matrices") else None),
        "hourly_profiles": (analyses["radar"] if analyses.get("radar") else None),
        "wordcloud": ({"overall": [{"word": w, "count": c} for w, c in
                                   analyses["wordcloud"]["overall"].most_common(50)],
                       "per_member": {m: [{"word": w, "count": c} for w, c in
                                          cnt.most_common(20)]
                                      for m, cnt in analyses["wordcloud"]["per_member"].items()}}
                      if analyses.get("wordcloud") else None),
        "monologues": ({"email_moments": analyses["monologues"]["email_moments"],
                        "longest_run_len": analyses["monologues"]["longest_run_len"],
                        "per_member_longest": dict(analyses["monologues"]["per_member_longest"])}
                       if analyses.get("monologues") else None),
        "unsent": dict(analyses["unsent"]) if analyses.get("unsent") else None,
        "taken_down": dict(analyses["taken_down"]) if analyses.get("taken_down") and any(analyses["taken_down"].values()) else None,
        "emojis": ({"total": analyses["emojis"]["total_emojis"],
                    "emojis_per_100": analyses["emojis"]["emojis_per_100"],
                    "per_member": {m: [{"emoji": e, "count": c} for e, c in cnt.most_common(20)]
                                   for m, cnt in analyses["emojis"]["per_member"].items()},
                    "per_year": {y: [{"emoji": e, "count": c} for e, c in cnt.most_common(top)]
                                 for y, cnt in analyses["emojis"]["per_year"].items()}}
                   if analyses.get("emojis") else None),
        "questions": ({"total": analyses["questions"]["total_questions"],
                       "answered": analyses["questions"]["total_answered"],
                       "unanswered": analyses["questions"]["unanswered_count"],
                       "table": analyses["questions"]["table"]}
                      if analyses.get("questions") else None),
        "topics": (analyses["topics"]["by_year"]
                   if analyses.get("topics") and analyses["topics"]["by_year"] else None),
        "running_jokes": ([{"phrase": j["phrase"], "count": j["count"],
                            "members": j["members"], "years": j["years"],
                            "example": j["example"]}
                           for j in analyses["jokes"]["jokes"]]
                          if analyses.get("jokes") and analyses["jokes"]["jokes"] else None),
    }
    if analyses.get("track"):
        payload["tracked_terms"] = {t: {"count": d["count"],
                                        "per_member": dict(d["per_member"]),
                                        "by_year": dict(d["by_year"])}
                                    for t, d in analyses["track"].items()}
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def console_summary(title, stats, analyses, dates):
    print(f"  {title}: {stats['total']:,} messages, {dates[0]} -> {dates[-1]}")
    top = stats["member_msgs"].most_common(3)
    print(f"  Top member: {top[0][0]} ({top[0][1]:,} msgs)")
    if top[1:]:
        print(f"  Runners-up: {', '.join(f'{n} ({c:,})' for n, c in top[1:])}")
    react = analyses["reactions"]
    if react["most_reactive"]:
        print(f"  Most reactive: {react['most_reactive']}")
    conv = analyses.get("conversations", {})
    if conv:
        print(f"  Conversations: {conv['conversation_count']:,} sessions (30-min gap), "
              f"longest {conv['longest_run_len']} msgs")
    ghosts = analyses.get("ghosting", {})
    if ghosts:
        worst = max(ghosts.items(), key=lambda kv: kv[1])
        print(f"  Longest silence: {worst[0]} ({worst[1]} days)")
    ex = analyses.get("extremes")
    if ex:
        print(f"  Record day: {ex['record_day']} ({ex['record_day_count']} msgs)")
    senti = analyses.get("sentiment")
    if senti:
        best = max(senti["per_member"].items(), key=lambda kv: kv[1])
        lowest = min(senti["per_member"].items(), key=lambda kv: kv[1])
        print(f"  Mood: {best[0]} {best[1]:+.2f} / {lowest[0]} {lowest[1]:+.2f}")
    swear = analyses["swear"]
    if swear["total_hits"]:
        print(f"  Swear messages: {swear['total_hits']}")
    mono = analyses.get("monologues")
    if mono and mono["per_member_longest"]:
        top_mono = mono["per_member_longest"].most_common(1)[0]
        print(f"  Longest solo run: {top_mono[0]} ({top_mono[1]} msgs in a row)")
    pm = analyses.get("pair_matrices")
    if pm:
        best_pair = max(((a, b, c) for a, row in pm["reply"].items()
                         for b, c in row.items()), key=lambda x: x[2], default=None)
        if best_pair:
            print(f"  Reply buddy: {best_pair[0]} -> {best_pair[1]} ({best_pair[2]}x)")
    unsent = analyses.get("unsent")
    if unsent and any(unsent.values()):
        print(f"  Unsent messages: {sum(unsent.values())}")
    emo = analyses.get("emojis")
    if emo and emo["per_year"]:
        top_emoji = _sum_counters(emo["per_year"].values()).most_common(1)
        if top_emoji:
            print(f"  Top emoji: {emoji_lib.demojize(top_emoji[0][0]).strip(':')} "
                  f"({top_emoji[0][1]}x)")
    qst = analyses.get("questions")
    if qst and qst["total_questions"]:
        print(f"  Questions: {qst['total_questions']} asked, "
              f"{100 * qst['total_answered'] // max(1, qst['total_questions'])}% answered")
    jokes = analyses.get("jokes")
    if jokes and jokes["jokes"]:
        print(f"  Running joke: \"{jokes['jokes'][0]['phrase']}\"")


def load_track_file(path):
    if not path:
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        print(f"  [warn] cannot read track file {path}")
        return []
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def process_thread(thread_dir, args):
    loaded = load_thread(thread_dir)
    if loaded is None:
        print(f"  [skip] {thread_dir}: no message files")
        return
    title, participants, raw = loaded
    # Release the raw export as it is converted: on a chat of a few hundred
    # thousand messages the raw dicts and the normalized ones are each of the
    # order of gigabytes, and holding both at once doubles the peak.
    loaded = None
    try:
        msgs = normalize_messages(raw, tz=args.tz, consume=True)
    except ValueError as exc:
        print(f"  [error] {thread_dir}: {exc}")
        return
    finally:
        raw = None
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

    # After anonymization, since that rewrites message content.
    add_derived_fields(msgs)

    _progress(args, "normalized")
    oldest = msgs[0]["dt"].strftime("%Y-%m-%d %H:%M")
    newest = msgs[-1]["dt"].strftime("%Y-%m-%d %H:%M")
    print(f"\n  Thread: {title}  ({len(msgs):,} messages, {oldest} -> {newest})")

    _progress(args, "core stats")
    skip = _parse_skip(args.skip)
    if skip:
        print(f"  Skipping: {', '.join(sorted(skip))}")
    stats = core_stats(msgs)
    track_terms = [t.strip() for t in args.track.split(",") if t.strip()] if args.track else []
    track_terms = track_terms + [t for t in load_track_file(args.track_file) if t not in track_terms]
    _progress(args, "analyses")
    analyses = {
        "yearly": yearly_recaps(msgs),
        "personality": personalities(msgs, top=args.top),
        "reactions": reaction_stats(msgs, top=args.top),
        "speed": response_speed(msgs, top=args.top),
        "swear": swear_stats(msgs),
        "track": custom_tracking(msgs, track_terms),
        "weird": weird_statements(msgs, top=args.top),
        "links_domains": links_domains(msgs, top=args.top),
        "media": media_leaderboard(msgs),
        "length_trends": length_trends(msgs),
        "word_trends": word_trends(msgs, top=args.top),
        "conversations": conversation_starters(msgs),
        "reply_chains": reply_chains(msgs, top=args.top),
        "ghosting": ghosting(msgs),
        "extremes": extremes(msgs),
        "heatmap": activity_heatmap(msgs),
        "pace": pace_trends(msgs),
        "pair_matrices": pair_matrices(msgs),
        "radar": hourly_radar(msgs),
        "monologues": monologues(msgs),
        "unsent": unsent_stats(msgs),
        "taken_down": taken_down_stats(msgs),
        "emojis": emoji_stats(msgs),
        "questions": question_stats(msgs),
    }
    # Skipped analyses are left out of the dict entirely rather than set to
    # None, so every reader's `analyses.get(name, {})` keeps working.
    if "wordcloud" not in skip:
        analyses["wordcloud"] = word_cloud_data(msgs)
    if "topics" not in skip:
        analyses["topics"] = topic_words(msgs)
    if "sentiment" not in skip:
        analyses["sentiment"] = sentiment_analysis(msgs)
    if "jokes" not in skip:
        _progress(args, "running jokes")
        analyses["jokes"] = inside_jokes(msgs)

    out_dir = Path(args.output) / _slug(title)
    _progress(args, "charts")
    write_charts(msgs, stats, analyses, out_dir, track_terms, top=args.top)
    _progress(args, "writing")
    write_summary(title, stats, analyses, analyses["track"], out_dir, anonymized,
                  [oldest[:10], newest[:10]], top=args.top)
    write_report_html(title, stats, analyses, out_dir, anonymized,
                      [oldest[:10], newest[:10]], top=args.top)
    _progress(args, "year reviews")
    write_year_reviews(title, stats, analyses, out_dir)
    if args.json:
        write_summary_json(title, stats, analyses, out_dir, anonymized,
                           [oldest[:10], newest[:10]], top=args.top)
    console_summary(title, stats, analyses, [oldest[:10], newest[:10]])
    if args.progress:
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()
    print(f"  Wrote output to {out_dir}")
    print(f"  First message: {oldest}  |  Last message: {newest}")
    if not args.anonymize:
        print("  Hint: re-run with --anonymize to strip names for sharing.")


def _parse_skip(value):
    if not value:
        return set()
    names = {n.strip().lower() for n in str(value).split(",") if n.strip()}
    for unknown in sorted(names - set(SKIPPABLE)):
        print(f"  [warn] unknown --skip value {unknown!r}; "
              f"known: {', '.join(SKIPPABLE)}")
    return names & set(SKIPPABLE)


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "thread"


def _progress(args, phase):
    if args.progress:
        sys.stderr.write(f"\r  [{phase}] analyzing...")
        sys.stderr.flush()


def serve(thread_dirs, args):
    from chat_ui import run_server
    threads = []
    for d in thread_dirs:
        loaded = load_thread(d)
        if loaded is None:
            continue
        title, participants, raw = loaded
        try:
            msgs = normalize_messages(raw, tz=args.tz)
        except ValueError as exc:
            print(f"  [error] {d}: {exc}")
            continue
        if not msgs:
            print(f"  [skip] {d}: no usable messages")
            continue
        if args.anonymize:
            apply_anonymization(msgs, anonymize_map(msgs))
        threads.append({"slug": _slug(title), "title": title,
                        "thread_dir": d, "msgs": msgs})
    if not threads:
        print("[error] no readable threads to serve")
        return 1
    print(f"\nStarting local chat reader on http://127.0.0.1:{args.port} "
          f"(localhost only)")
    run_server(threads, args.port, args.output)
    return 0


def _message_has_content(m):
    if m.get("content"):
        return True
    if any(m.get(k) for k in ("photos", "sticker", "gifs", "videos", "audio_files", "files")):
        return True
    share = m.get("share") if isinstance(m.get("share"), dict) else None
    if share and (share.get("link") or share.get("text")):
        return True
    if m.get("call_duration") or m.get("polls") or m.get("reactions"):
        return True
    return False


def _message_media_uris(m):
    for key in ("photos", "gifs", "videos", "audio_files"):
        for item in m.get(key) or []:
            if isinstance(item, dict) and item.get("uri"):
                yield item["uri"]
    for item in m.get("files") or []:
        if isinstance(item, dict) and item.get("uri"):
            yield item["uri"]


def resolve_media_path(thread_dir, uri):
    """Resolve an export's media uri to a file inside the thread folder.

    Real Messenger exports store uris relative to the *export root*, like
    "your_facebook_activity/messages/inbox/<thread>/photos/x.jpg", while the
    tool is pointed at the thread folder itself. Leading path segments are
    dropped one at a time until the tail lands on a real file, which resolves
    root-relative and thread-relative uris alike, and keeps working if the
    thread folder has been renamed.

    The result must still sit inside the thread folder, so an export naming a
    path outside it resolves to nothing rather than being read.
    """
    base = thread_dir.resolve()
    parts = [p for p in uri.replace("\\", "/").split("/") if p and p != "."]
    for i in range(len(parts)):
        p = base.joinpath(*parts[i:]).resolve()
        if p.is_file() and base in p.parents:
            return p
    return None


def check_thread(thread_dir, args):
    files = sorted(thread_dir.glob("message_*.json"), key=numeric_key)
    if not files:
        print(f"  [check] {thread_dir}: no message files")
        return
    title = thread_dir.name
    types = Counter()
    keys = Counter()
    empty = 0
    missing_media = []
    media_count = 0
    total = 0
    unreadable = 0
    dupes = 0
    seen_ids = set()
    seen_fallback = set()
    per_file = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            raw = re.sub(r"[\x00-\x1f\x7f]", " ", f.read_text(encoding="utf-8", errors="ignore"))
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [check] {f.name}: unreadable ({exc})")
                unreadable += 1
                continue
        if data.get("title"):
            title = data["title"]
        f_total = 0
        f_oldest = f_newest = None
        for m in data.get("messages", []):
            if not isinstance(m, dict):
                continue
            total += 1
            f_total += 1
            ts = m.get("timestamp_ms")
            if ts is None:
                ts = m.get("timestamp")
            if ts is not None:
                try:
                    dt = datetime.fromtimestamp(int(float(ts)) / 1000)
                except (TypeError, ValueError):
                    dt = None
                if dt is not None:
                    f_oldest = dt if f_oldest is None else min(f_oldest, dt)
                    f_newest = dt if f_newest is None else max(f_newest, dt)
            types[m.get("type", "Generic")] += 1
            for k in m:
                keys[k] += 1
            mid = m.get("id")
            if mid is not None:
                if mid in seen_ids:
                    dupes += 1
                else:
                    seen_ids.add(mid)
            else:
                fkey = (m.get("sender_name"), m.get("timestamp_ms") or m.get("timestamp"),
                        m.get("content"))
                if fkey in seen_fallback:
                    dupes += 1
                else:
                    seen_fallback.add(fkey)
            if not _message_has_content(m):
                empty += 1
            for uri in _message_media_uris(m):
                media_count += 1
                if resolve_media_path(thread_dir, uri) is None:
                    missing_media.append((f.name, uri))
        per_file.append({
            "file": f.name, "messages": f_total,
            "oldest": f_oldest.strftime("%Y-%m-%d %H:%M") if f_oldest else None,
            "newest": f_newest.strftime("%Y-%m-%d %H:%M") if f_newest else None,
        })

    gaps = []
    for a, b in zip(per_file, per_file[1:]):
        if a["newest"] and b["oldest"]:
            days = (datetime.strptime(b["oldest"], "%Y-%m-%d %H:%M")
                    - datetime.strptime(a["newest"], "%Y-%m-%d %H:%M")).days
            gaps.append((a["file"], b["file"], days))

    unknown_types = sorted(t for t in types if t not in KNOWN_TYPES)
    unknown_keys = sorted(k for k in keys if k not in KNOWN_MESSAGE_KEYS)

    print(f"  [check] {title}  ({thread_dir})")
    print(f"    files: {len(files)} | messages: {total:,} | unreadable: {unreadable:,}")
    print(f"    types: " + ", ".join(f"{t} {types[t]:,}" for t in sorted(types)) or "none")
    if unknown_types:
        print(f"    unknown types: {', '.join(unknown_types)}")
    if unknown_keys:
        print(f"    unknown message keys: {', '.join(unknown_keys)}")
    print(f"    empty messages: {empty:,}")
    print(f"    media attachments: {media_count:,} | missing on disk: {len(missing_media):,}")
    for fname, uri in missing_media[:5]:
        print(f"      missing: {fname} -> {uri}")
    print(f"    duplicate messages: {dupes:,}")
    big_gaps = [g for g in gaps if g[2] > 90]
    if big_gaps:
        print("    file gaps over 90 days:")
        for a, b, days in big_gaps:
            print(f"      {a} -> {b}: {days} days")
    else:
        print("    file gaps: none over 90 days")

    if args.json:
        out_dir = Path(args.output) / _slug(title)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "title": title,
            "thread_dir": str(thread_dir),
            "files": per_file,
            "messages": total,
            "unreadable": unreadable,
            "types": dict(types),
            "unknown_types": unknown_types,
            "unknown_keys": unknown_keys,
            "empty_messages": empty,
            "media_count": media_count,
            "missing_media": [{"file": f, "uri": u} for f, u in missing_media[:100]],
            "duplicates": dupes,
            "file_gaps_over_90d": [{"from": a, "to": b, "days": d} for a, b, d in big_gaps],
        }
        (out_dir / "check.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    wrote {out_dir / 'check.json'}")


def _thread_fingerprint(thread_dir):
    files = sorted(thread_dir.glob("message_*.json"), key=numeric_key)
    return [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def _config_signature(args):
    # Tracked terms are resolved, not just named: editing a --track-file has to
    # invalidate the cache the same way editing --track does.
    terms = sorted(t.strip() for t in args.track.split(",") if t.strip()) if args.track else []
    terms += [t for t in load_track_file(args.track_file) if t not in terms]
    return json.dumps({
        "year": args.year, "anonymize": args.anonymize, "top": args.top,
        "track": sorted(terms),
        "tz": args.tz or "",
        "skip": sorted(_parse_skip(args.skip)),
    }, sort_keys=True)


def run(args):
    thread_dirs = find_thread_dirs(args.input)
    if not thread_dirs:
        return 1
    if args.check:
        for d in thread_dirs:
            check_thread(d, args)
        return 0
    if args.serve:
        return serve(thread_dirs, args)
    if len(thread_dirs) > 1:
        print(f"Found {len(thread_dirs)} threads:")
        for d in thread_dirs[:20]:
            print(f"  - {d}")
        print("Processing all threads...")
    state_path = Path(args.output) / ".chatflashback_state.json"
    state = {}
    if args.incremental and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    cfg_sig = _config_signature(args)
    for d in thread_dirs:
        if args.incremental:
            key = str(Path(d).resolve())
            prev = state.get(key)
            if isinstance(prev, dict) and prev.get("cfg") == cfg_sig \
                    and prev.get("fp") == _thread_fingerprint(d):
                print(f"  [skip] {d}: unchanged since last run")
                continue
            process_thread(d, args)
            state[key] = {"fp": _thread_fingerprint(d), "cfg": cfg_sig}
        else:
            process_thread(d, args)
    if args.incremental:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return 0


def _join_tz_arg(argv):
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--tz",) and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append(tok + "=" + argv[i + 1])
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(argv=None):
    argv = _join_tz_arg(list(sys.argv[1:]) if argv is None else list(argv))
    parser = argparse.ArgumentParser(
        prog="analyze_chat.py",
        description=(
            "Analyze a Facebook Messenger export and generate flashback analytics: "
            "yearly recaps, member personalities, reaction dynamics, response-speed "
            "leaderboards, swear-word stats, custom term tracking, conversation "
            "starters, reply chains, ghosting stats, sentiment (VADER), a 'weirdest "
            "statements' highlight reel, and a self-contained report.html. Runs 100% locally."
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
    parser.add_argument("--track-file", default="",
                        help="File with tracked terms, one per line (# comments and blank lines ignored)")
    parser.add_argument("--year", type=int,
                        help="Limit the analysis to a single year, e.g. --year 2017")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of entries in leaderboards (default: 10)")
    parser.add_argument("--json", action="store_true",
                        help="Also write summary.json with the same data as the report")
    parser.add_argument("--serve", action="store_true",
                        help="Start the local chat reader web UI instead of writing reports")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for --serve (default: 8080)")
    parser.add_argument("--tz", default="",
                        help="Timezone for analysis, e.g. +03:00 or America/New_York "
                             "(Messenger timestamps are UTC; default is your system timezone)")
    parser.add_argument("--config", default="",
                        help="JSON config file with any of the CLI options")
    parser.add_argument("--skip", default="",
                        help="Comma-separated analyses to skip: "
                             + ", ".join(SKIPPABLE)
                             + ". 'jokes' uses the most memory and 'sentiment' "
                               "is the slowest, so skip those on a very large chat")
    parser.add_argument("--progress", action="store_true",
                        help="Show phase progress while analyzing")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip threads that are unchanged since the last run")
    parser.add_argument("--check", action="store_true",
                        help="Validate the export instead of analyzing: report unknown message "
                             "types/keys, empty messages, missing media files, duplicate "
                             "messages, and gaps between message files (always exits 0)")
    pre = parser.parse_args(argv)
    if pre.config:
        try:
            cfg = json.loads(Path(pre.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [warn] cannot read config {pre.config}: {exc}")
            cfg = {}
        if isinstance(cfg, dict):
            parser.set_defaults(**{str(k).replace("-", "_"): v for k, v in cfg.items()})
        else:
            print(f"  [warn] config {pre.config} must be a JSON object")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
