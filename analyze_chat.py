#!/usr/bin/env python3
"""chat-flashback: turn a Messenger export into a report about a group chat.

Parses a Facebook Messenger JSON export (or a WhatsApp/Telegram folder), computes
the analytics, and writes charts plus a summary.md. Everything runs locally.
"""

import argparse
import base64
import html as html_lib
import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta
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
REPLY_WINDOW_SECONDS = 60 * 60
CONVERSATION_WINDOW_SECONDS = 30 * 60
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
            "id": m.get("id"),
            "sender": sender,
            "ts_ms": ts_ms,
            "dt": datetime.fromtimestamp(ts_ms / 1000),
            "content": content,
            "mtype": m.get("type", "Generic"),
            "reactions": reactions,
            "has_photo": bool(m.get("photos")),
            "has_sticker": bool(m.get("sticker")),
            "photo_uris": [p.get("uri") for p in (m.get("photos") or []) if p.get("uri")],
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
    patterns = [(re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE), label)
                for n, label in mapping.items()]
    for m in msgs:
        m["sender"] = mapping.get(m["sender"], "Person ?")
        m["reactions"] = [(mapping.get(a, "Person ?"), r) for a, r in m["reactions"]]
        content = m["content"]
        if content:
            for pattern, label in patterns:
                content = pattern.sub(label, content)
            m["content"] = content
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
    for m in msgs:
        if m["has_photo"]:
            photos[m["sender"]] += 1
        if m["has_sticker"]:
            stickers[m["sender"]] += 1
    return {"photos": photos, "stickers": stickers}


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
        for w in tokenize(m["content"]):
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
        if m["has_photo"] or m["has_sticker"]:
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
        for w in tokenize(m["content"]):
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

    # top domains
    ld = analyses.get("links_domains", {})
    if ld and ld["domains"]:
        top_domains = ld["domains"].most_common(10)
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            _bar(fig, ax, [d for d, _ in top_domains], [c for _, c in top_domains],
                 "Top domains shared")
            save(fig, "top_domains.png")

    # media leaderboard
    media = analyses.get("media", {})
    if media and (media["photos"] or media["stickers"]):
        members = sorted(set(media["photos"]) | set(media["stickers"]))
        photos = [media["photos"].get(m, 0) for m in members]
        stickers = [media["stickers"].get(m, 0) for m in members]
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = range(len(members))
            ax.bar(x, photos, width=0.4, label="Photos", color=PALETTE[1])
            ax.bar([i + 0.4 for i in x], stickers, width=0.4, label="Stickers", color=PALETTE[3])
            ax.set_xticks([i + 0.2 for i in x])
            ax.set_xticklabels(members, rotation=30, ha="right", fontsize=9)
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
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, (word, series) in enumerate(sorted(wt.items())):
                ax.plot(list(series), list(series.values()), marker="o", label=word,
                        color=PALETTE[i % len(PALETTE)])
            ax.set_xticks(list(range(len(next(iter(wt.values()))))))
            ax.set_xticklabels(list(next(iter(wt.values()))), rotation=45, ha="right", fontsize=8)
            ax.set_title("Top words over time", fontweight="bold")
            ax.legend(fontsize=8)
            save(fig, "word_trends.png")

    # conversation starters
    conv = analyses.get("conversations", {})
    if conv and conv["starters"]:
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_starters = conv["starters"].most_common(10)
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
            top_ghosts = sorted(ghosts.items(), key=lambda kv: -kv[1])[:10]
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
            rd_month = ex["record_day"].strftime("%Y-%m")
            if rd_month in months:
                idx = months.index(rd_month)
                peak = max(vals)
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
            top_senti = sorted(senti["per_member"].items(), key=lambda kv: -kv[1])[:10]
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
            ax.set_title("Media (photos/stickers) per year", fontweight="bold")
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
            top_mono = mono["per_member_longest"].most_common(10)
            _bar(fig, ax, [n for n, _ in top_mono], [c for _, c in top_mono],
                 "Longest solo run (messages in a row)")
            save(fig, "monologues.png")

    # unsent
    unsent = analyses.get("unsent")
    if unsent and any(unsent.values()):
        with plt.rc_context(theme):
            fig, ax = plt.subplots(figsize=(8, 4))
            top_unsent = unsent.most_common(10)
            _bar(fig, ax, [n for n, _ in top_unsent], [c for _, c in top_unsent],
                 "Unsent messages per member")
            save(fig, "unsent.png")


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
    lines.append(f"- **Longest daily streak**: {stats['longest_streak']} day{'s' if stats['longest_streak'] != 1 else ''}")
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

    ld = analyses.get("links_domains", {})
    if ld and ld["domains"]:
        lines.append("## Links and domains")
        lines.append("")
        lines.append("| Domain | Shares |")
        lines.append("|---|---|")
        for domain, count in ld["domains"].most_common(10):
            lines.append(f"| {domain} | {count} |")
        lines.append("")

    media = analyses.get("media", {})
    if media and (media["photos"] or media["stickers"]):
        lines.append("## Media leaderboard")
        lines.append("")
        lines.append("| Member | Photos | Stickers |")
        lines.append("|---|---|---|")
        for member in sorted(set(media["photos"]) | set(media["stickers"]),
                             key=lambda m: -(media["photos"].get(m, 0) + media["stickers"].get(m, 0))):
            lines.append(f"| {member} | {media['photos'].get(member, 0)} | "
                         f"{media['stickers'].get(member, 0)} |")
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
        for member, count in conv["starters"].most_common(10):
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
                elif m["has_photo"]:
                    labels.append(f"**{m['sender']}**: [photo]")
                elif m["has_sticker"]:
                    labels.append(f"**{m['sender']}**: [sticker]")
                else:
                    labels.append(f"**{m['sender']}**: [media]")
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
                         f"({peak_media[1]} photos/stickers).")
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
        for member, length in mono["per_member_longest"].most_common(10):
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
        for member, count in unsent.most_common(10):
            lines.append(f"| {member} | {count} |")
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
    speed = analyses.get("speed", {})
    if speed["table"]:
        fastest = speed["table"][0]
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
}


def _sec(sid, title, inner):
    return f'<section id="{sid}"><h2>{title}</h2>{inner}</section>'


def _thead(columns):
    return "<thead><tr>" + "".join(f"<th>{html_lib.escape(c)}</th>" for c in columns) + "</tr></thead>"


def _table(columns, rows):
    body = "<tbody>" + "".join("<tr>" + "".join(f"<td>{r}</td>" for r in row) + "</tr>"
                               for row in rows) + "</tbody>"
    return f"<table>{_thead(columns)}{body}</table>"


def write_report_html(title, stats, analyses, out_dir, anonymized, dates):
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
                   for m, c in stats["member_msgs"].most_common(10)]
    reactor_rows = [(html_lib.escape(m), f"{c:,}") for m, c in react["reactor"].most_common(10)]

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
                for m, n in mono["per_member_longest"].most_common(10)]
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
    sections.append(_sec("charts", "Charts", imgs))

    nav = "".join(
        f'<a href="#{sid}">{html_lib.escape(label)}</a>'
        for sid, label in [("highlights", "Highlights"), ("leaderboard", "Leaderboard"),
                           ("reactive", "Reactive"), ("pair_dynamics", "Pairs"),
                           ("monologues", "Monologues"), ("weirdest", "Weirdest"),
                           ("extremes", "Extremes"), ("sentiment", "Sentiment"),
                           ("charts", "Charts")]
    )
    body = f"""<div class="topbar"><span class="brand">{html_lib.escape(title)} flashback</span>
<nav>{nav}</nav>
<button id="theme" title="Toggle theme" aria-label="Toggle theme">Dark</button></div>
<main>
<h1>{html_lib.escape(title)} flashback</h1>
<p class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
{"(names anonymized)" if anonymized else ""}</p>
<div class="cards">
<div class="card"><b>{stats['total']:,}</b><span>messages</span></div>
<div class="card"><b>{len(stats['member_msgs'])}</b><span>members</span></div>
<div class="card"><b>{dates[0]}</b><span>start</span></div>
<div class="card"><b>{dates[-1]}</b><span>end</span></div>
<div class="card"><b>{stats['longest_streak']}</b><span>day streak</span></div>
<div class="card"><b>{stats['media']}</b><span>media</span></div>
<div class="card"><b>{stats['calls']}</b><span>calls</span></div>
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
#theme{{margin-left:auto;border:1px solid var(--border);background:var(--bg);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}}
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


def write_summary_json(title, stats, analyses, out_dir, anonymized, dates):
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
        "members": len(stats["member_msgs"]),
        "longest_streak_days": stats["longest_streak"],
        "media": stats["media"],
        "calls": stats["calls"],
        "leaderboard": [{"member": m, "messages": c}
                        for m, c in stats["member_msgs"].most_common(10)],
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
            "reactors": [{"member": m, "count": c} for m, c in react["reactor"].most_common(10)],
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
                           for m, c in analyses["swear"]["member_hits"].most_common(10)],
        },
        "weirdest_statements": [
            {"member": s["member"], "ts": s["dt"].strftime("%Y-%m-%d %H:%M"),
             "score": s["score"], "reasons": s["reasons"], "snippet": s["snippet"]}
            for s in analyses["weird"]
        ],
        "links": {
            "top_domains": [{"domain": d, "count": c}
                            for d, c in analyses.get("links_domains", {}).get("domains", {}).most_common(10)],
            "top_links": [{"url": u, "count": c}
                          for u, c in analyses.get("links_domains", {}).get("links", {}).most_common(10)],
        },
        "media": {m: {"photos": analyses["media"]["photos"].get(m, 0),
                      "stickers": analyses["media"]["stickers"].get(m, 0)}
                  for m in sorted(set(analyses["media"]["photos"]) | set(analyses["media"]["stickers"]))},
        "conversations": {
            "count": analyses["conversations"]["conversation_count"],
            "longest_run_msgs": analyses["conversations"]["longest_run_len"],
            "starters": [{"member": m, "count": c}
                         for m, c in analyses["conversations"]["starters"].most_common(10)],
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
    track_terms = track_terms + [t for t in load_track_file(args.track_file) if t not in track_terms]
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
        "sentiment": sentiment_analysis(msgs),
        "heatmap": activity_heatmap(msgs),
        "pace": pace_trends(msgs),
        "pair_matrices": pair_matrices(msgs),
        "radar": hourly_radar(msgs),
        "wordcloud": word_cloud_data(msgs),
        "monologues": monologues(msgs),
        "unsent": unsent_stats(msgs),
    }

    out_dir = Path(args.output) / _slug(title)
    write_charts(msgs, stats, analyses, out_dir, track_terms)
    write_summary(title, stats, analyses, analyses["track"], out_dir, anonymized,
                  [oldest[:10], newest[:10]])
    write_report_html(title, stats, analyses, out_dir, anonymized, [oldest[:10], newest[:10]])
    if args.json:
        write_summary_json(title, stats, analyses, out_dir, anonymized, [oldest[:10], newest[:10]])
    console_summary(title, stats, analyses, [oldest[:10], newest[:10]])
    print(f"  Wrote output to {out_dir}")
    print(f"  First message: {oldest}  |  Last message: {newest}")
    if not args.anonymize:
        print("  Hint: re-run with --anonymize to strip names for sharing.")


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "thread"


def serve(thread_dirs, args):
    from chat_ui import run_server
    threads = []
    for d in thread_dirs:
        loaded = load_thread(d)
        if loaded is None:
            continue
        title, participants, raw = loaded
        msgs = normalize_messages(raw)
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


def run(args):
    thread_dirs = find_thread_dirs(args.input)
    if not thread_dirs:
        return 1
    if args.serve:
        return serve(thread_dirs, args)
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
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
