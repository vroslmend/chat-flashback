"""chat-flashback reader: a local web UI to browse a parsed chat.

Started with `python analyze_chat.py --input <thread> --serve`. Binds to
127.0.0.1 only and serves a Messenger-style reader: day-grouped feed, member
filters, full-text search, reply threading, media, and sentiment tinting.
"""

import bisect
import json
import mimetypes
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import analyze_chat as ac

_PAGE_SIZE = 400


def _snippet(text, n=90):
    if not text:
        return "(no text)"
    text = text.replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


class ThreadIndex:
    def __init__(self, slug, title, thread_dir, msgs):
        self.slug = slug
        self.title = title
        self.thread_dir = Path(thread_dir)
        self.msgs = msgs
        self.by_id = {m.get("id"): m for m in msgs if m.get("id") is not None}
        self.colors = {}
        members = sorted({m["sender"] for m in msgs})
        for i, name in enumerate(members):
            self.colors[name] = ac.PALETTE[i % len(ac.PALETTE)]
        self.all_pairs = []
        self.member_pairs = {}
        # (month, day) -> indices, so "on this day" can group by the same local
        # calendar date the rest of the report uses. Rebuilding the day window
        # from timestamps would apply the system timezone instead of --tz.
        self.by_monthday = {}
        for i, m in enumerate(msgs):
            self.all_pairs.append((m["ts_ms"], i))
            self.member_pairs.setdefault(m["sender"], []).append((m["ts_ms"], i))
            dt = m["dt"]
            self.by_monthday.setdefault((dt.month, dt.day), []).append(i)
        self._sent_cache = {}

    def to_json(self, idx):
        m = self.msgs[idx]
        j = {
            "ts": m["ts_ms"], "sender": m["sender"], "color": self.colors[m["sender"]],
            "content": m["content"], "mtype": m["mtype"],
            "reactions": [{"actor": a, "reaction": r} for a, r in m["reactions"]],
            "has_photo": m.get("has_photo", False), "photo_uris": m.get("photo_uris", []),
            "has_sticker": m.get("has_sticker", False),
            "has_gif": m.get("has_gif", False), "gif_uris": m.get("gif_uris", []),
            "has_video": m.get("has_video", False), "video_uris": m.get("video_uris", []),
            "has_audio": m.get("has_audio", False), "audio_uris": m.get("audio_uris", []),
            "has_file": m.get("has_file", False), "file_uris": m.get("file_uris", []),
            "file_names": m.get("file_names", []),
            "is_taken_down": m.get("is_taken_down", False),
            "link": m.get("link"),
            "is_unsent": m.get("is_unsent", False), "reply_to": None, "sentiment": None,
        }
        rid = m.get("reply_to")
        if rid is not None and rid in self.by_id:
            p = self.by_id[rid]
            j["reply_to"] = {"sender": p["sender"], "snippet": _snippet(p["content"])}
        if ac._VADER is not None and m["content"]:
            c = self._sent_cache.get(m["content"])
            if c is None:
                c = ac._VADER.polarity_scores(m["content"])["compound"]
                self._sent_cache[m["content"]] = c
                if len(self._sent_cache) > 50_000:
                    self._sent_cache.clear()
            j["sentiment"] = c
        return j

    def meta(self):
        total = len(self.msgs)
        member_counts = {}
        for m in self.msgs:
            member_counts[m["sender"]] = member_counts.get(m["sender"], 0) + 1
        members = [{"name": n, "count": c, "color": self.colors[n]}
                   for n, c in sorted(member_counts.items(), key=lambda kv: -kv[1])]
        return {
            "title": self.title,
            "slug": self.slug,
            "total": total,
            "start": self.msgs[0]["ts_ms"],
            "end": self.msgs[-1]["ts_ms"],
            "members": members,
            "sentiment_available": ac._VADER is not None,
            "has_replies": bool(self.by_id),
        }

    def page(self, before=None, after=None, member=None, q=None, limit=_PAGE_SIZE, regex=False):
        if q:
            return self._search(q, member, limit, regex)
        pairs = self.member_pairs.get(member) if member else self.all_pairs
        n = len(pairs)
        if before is not None:
            end = bisect.bisect_left(pairs, (before, -1))
            start = max(0, end - limit)
            sel = pairs[start:end][::-1]
            next_before = pairs[start][0] if start > 0 else None
            next_after = None
        elif after is not None:
            start = bisect.bisect_right(pairs, (after, 10 ** 15))
            end = min(n, start + limit)
            sel = pairs[start:end]
            next_after = pairs[end - 1][0] if end < n else None
            next_before = None
        else:
            end = n
            start = max(0, end - limit)
            sel = pairs[start:end][::-1]
            next_before = pairs[start][0] if start > 0 else None
            next_after = None
        return {"messages": [self.to_json(i) for _, i in sel],
                "next_before": next_before, "next_after": next_after,
                "search": False}

    def _search(self, q, member, limit, regex=False):
        ql = q.lower()
        pattern = None
        if regex:
            try:
                pattern = re.compile(q, re.IGNORECASE)
            except re.error:
                pattern = None
        # Keep indices, never the message dicts: recovering an index later with
        # msgs.index() is a linear scan of dict comparisons per hit, which turns
        # a search over a long chat into seconds of quadratic work.
        hits = []
        total = 0
        for i, m in enumerate(self.msgs):
            if member and m["sender"] != member:
                continue
            content = m["content"] or ""
            if pattern is not None:
                found = pattern.search(content) is not None
            else:
                found = ql in content.lower()
            if found:
                total += 1
                if len(hits) < limit:
                    hits.append(i)
        return {"messages": [self.to_json(i) for i in hits],
                "next_before": None, "next_after": None,
                "search": True, "total_matches": total,
                "shown": len(hits), "truncated": total > len(hits)}

    def day(self, month, day, limit=_PAGE_SIZE):
        out = sorted(self.by_monthday.get((month, day), []),
                     key=lambda i: self.msgs[i]["ts_ms"])
        years = sorted({self.msgs[i]["dt"].year for i in out})
        return {"messages": [self.to_json(i) for i in out[:limit]],
                "total": len(out), "years": years}

    def random_memory(self):
        reacted = [i for i, m in enumerate(self.msgs) if m["reactions"]]
        long_text = [i for i, m in enumerate(self.msgs)
                     if m["content"] and len(m["content"]) > 40]
        pool = reacted or long_text or list(range(len(self.msgs)))
        idx = random.choice(pool)
        return {"message": self.to_json(idx)}

    def resolve_media(self, rel):
        # Same resolver the analyzer uses, so the reader and --check agree on
        # which attachments exist and neither can drift out of the thread dir.
        return ac.resolve_media_path(self.thread_dir, rel)


def _read_int(params, key):
    v = params.get(key)
    if not v:
        return None
    try:
        return int(float(v[0]))
    except (TypeError, ValueError):
        return None


def make_handler(threads, output_dir):

    class Handler(BaseHTTPRequestHandler):
        server_version = "chat-flashback/1.0"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, path, ctype):
            """Stream a media file, honouring a single Range request.

            Streamed in chunks rather than read whole so a large video does not
            have to fit in memory, and Range is answered so players can seek.
            Anything that is not an image, video or audio is forced to download
            instead of rendering: an export can contain .html or .svg
            attachments, which would otherwise run as script on this origin.
            """
            size = path.stat().st_size
            start, end = 0, size - 1
            partial = False
            rng = self.headers.get("Range", "")
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
            if m and size:
                lo, hi = m.group(1), m.group(2)
                if lo:
                    start = min(int(lo), size - 1)
                    end = min(int(hi), size - 1) if hi else size - 1
                elif hi:                      # suffix range: last N bytes
                    start = max(0, size - int(hi))
                if start <= end:
                    partial = True
                else:
                    start, end = 0, size - 1
            length = end - start + 1
            inline = ctype.split("/")[0] in ("image", "video", "audio")
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if not inline:
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{path.name}"')
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            query = parse_qs(parts.query)
            if path == "/":
                return self._landing()
            if path == "/favicon.ico":
                return self._send(404, "")
            rest = path.split("/")[1:]
            if rest and rest[0] == "t" and len(rest) >= 2:
                slug = rest[1]
                thread = threads.get(slug)
                if thread is None:
                    return self._send(404, "not found")
                sub = [s for s in rest[2:] if s]
                if not sub:
                    # The title comes from the export, so it is untrusted text:
                    # escape it. The slug is safe by construction (ac._slug
                    # keeps only alphanumerics and underscores).
                    return self._send(200, viewer_html().replace("__SLUG__", slug)
                                      .replace("__TITLE__", html(thread.title)))
                if sub[0] == "api":
                    if len(sub) >= 2 and sub[1] == "messages":
                        return self._messages(thread, query)
                    if len(sub) >= 2 and sub[1] == "day":
                        return self._day(thread, query)
                    if len(sub) >= 2 and sub[1] == "random":
                        return self._json(thread.random_memory())
                    return self._json(thread.meta())
                if sub[0] == "report.html":
                    report = output_dir / slug / "report.html"
                    if report.is_file():
                        return self._send(200, report.read_bytes(), "text/html; charset=utf-8")
                    return self._send(404, "no report yet")
                if sub[0] == "year_in_review.html":
                    index = output_dir / slug / "year_in_review.html"
                    if index.is_file():
                        return self._send(200, index.read_bytes(), "text/html; charset=utf-8")
                    return self._send(404, "no year review yet")
                if sub[0] == "year" and len(sub) >= 2:
                    page = output_dir / slug / f"year_{sub[1]}.html"
                    if page.is_file():
                        return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                    return self._send(404, "no such year")
                if sub[0] == "media":
                    rel = "/".join(sub[1:])
                    media = thread.resolve_media(rel)
                    if media is None:
                        return self._send(404, "not found")
                    ctype = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
                    return self._send_media(media, ctype)
            return self._send(404, "not found")

        def _landing(self):
            items = "".join(
                f'<li><a href="/t/{t.slug}/">{html(t.title)}</a> '
                f"<span class=\"muted\">{len(t.msgs):,} messages</span></li>"
                for t in threads.values()
            )
            html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>chat-flashback</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#e8e8e8;background:#17191f}}
h1{{font-size:24px}} a{{color:#5b8ff9}} li{{margin:10px 0;font-size:15px}}
.muted{{color:#9aa0a6;font-size:13px}}
</style></head><body>
<h1>chat-flashback</h1>
<p class="muted">Local reader. Click a thread to open it. This server only listens on
your machine.</p>
<ul>{items}</ul>
</body></html>"""
            return self._send(200, html_doc)

        def _messages(self, thread, query):
            before = _read_int(query, "before")
            after = _read_int(query, "after")
            member = (query.get("member") or [None])[0]
            q = (query.get("q") or [None])[0]
            regex = (query.get("re") or [""])[0] == "1"
            limit = _read_int(query, "limit") or _PAGE_SIZE
            return self._json(thread.page(before=before, after=after,
                                          member=member or None, q=q or None,
                                          limit=limit, regex=regex))

        def _day(self, thread, query):
            date = (query.get("date") or [""])[0]
            try:
                month, day = int(date[5:7]), int(date[8:10])
            except (ValueError, IndexError):
                return self._send(400, "date must be YYYY-MM-DD")
            return self._json(thread.day(month, day))

    return Handler


def html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def viewer_html():
    return _VIEWER


_VIEWER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ | chat-flashback</title>
<style>
:root{--bg:#17191f;--panel:#20242d;--border:#2a2f3a;--fg:#e8e8e8;--muted:#9aa0a6;--me:#5b8ff9;--hover:rgba(255,255,255,0.04);--quote:rgba(255,255,255,0.03)}
[data-theme="light"]{--bg:#f7f8fa;--panel:#ffffff;--border:#e3e5e8;--fg:#202124;--muted:#777;--hover:rgba(0,0,0,0.04);--quote:rgba(0,0,0,0.02)}
*{box-sizing:border-box}
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg);line-height:1.45}
.topbar{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--border)}
.topbar h1{font-size:16px;margin:0 12px 0 0}
.topbar input,.topbar select,.topbar button{padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--fg);font-size:13px}
#q{flex:1;min-width:160px}
button{cursor:pointer}
button.active{background:var(--me);color:#fff;border-color:var(--me)}
.count{color:var(--muted);font-size:12px}
main{max-width:760px;margin:0 auto;padding:0 16px 120px}
.day{position:sticky;top:52px;z-index:10;display:flex;align-items:center;gap:10px;margin:22px 0 8px;font-size:12px;color:var(--muted)}
.day::before,.day::after{content:"";flex:1;height:1px;background:var(--border)}
.msg{display:flex;gap:10px;margin:2px 0;padding:4px 10px;border-radius:10px}
.msg:hover{background:var(--hover)}
.dot{width:30px;height:30px;border-radius:50%;flex:0 0 30px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;color:#111}
.body{flex:1;min-width:0}
.sender{font-size:12px;font-weight:700;margin-bottom:2px}
.text{font-size:14px;word-wrap:break-word;white-space:pre-wrap}
.time{font-size:11px;color:var(--muted);margin-left:6px;font-weight:400}
.quote{border-left:3px solid var(--border);padding:2px 8px;margin:2px 0 4px;font-size:12px;color:var(--muted);background:var(--quote);border-radius:4px}
.badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:8px;background:#3d2c5a;color:#c9b6ff;margin-left:6px}
[data-theme="light"] .badge{background:#e8e0ff;color:#5b3fa8}
.unsent{text-decoration:line-through;opacity:.6}
.media{margin:6px 0}
.media img{max-width:220px;max-height:220px;border-radius:10px;border:1px solid var(--border);display:block}
.media video{max-width:280px;max-height:220px;border-radius:10px;border:1px solid var(--border);display:block}
.media audio{width:280px;max-width:100%}
.media a{color:#5b8ff9;word-break:break-all;font-size:13px}
.reactions{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.ra{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1px 8px;font-size:12px}
.link{font-size:12px;color:#5b8ff9}
.link a{color:#5b8ff9;word-break:break-all}
.call{font-size:12px;color:var(--muted)}
.sent-pos{background:rgba(90,216,166,0.06)}
.sent-neg{background:rgba(232,104,74,0.06)}
.loader{text-align:center;color:var(--muted);padding:24px;font-size:13px}
.empty{text-align:center;color:var(--muted);padding:40px;font-size:14px}
</style></head><body>
<div class="topbar">
<h1>__TITLE__</h1>
<input id="q" placeholder="Search messages..."/>
<button id="re" title="Toggle regex search" aria-label="Toggle regex search">.*</button>
<select id="member"><option value="">Everyone</option></select>
<select id="order">
<option value="newest">Newest first</option>
<option value="oldest">Oldest first</option>
</select>
<input id="jump" type="date" title="Jump to date"/>
<button id="oday" title="Show this day in every year">On this day</button>
<button id="surprise" title="Random memory">Surprise me</button>
<button id="theme" title="Toggle theme" aria-label="Toggle theme">Light</button>
<span class="count" id="count"></span>
<a href="report.html" style="color:#5b8ff9;font-size:13px">Report</a>
<a href="year_in_review.html" style="color:#5b8ff9;font-size:13px">Years</a>
</div>
<main id="feed"></main>
<div class="loader" id="loader">Loading...</div>
<script>
var SLUG='__SLUG__';
var feed=document.getElementById('feed'), loader=document.getElementById('loader');
var qEl=document.getElementById('q'), memberEl=document.getElementById('member');
var orderEl=document.getElementById('order'), jumpEl=document.getElementById('jump');
var countEl=document.getElementById('count');
var reBtn=document.getElementById('re'), themeBtn=document.getElementById('theme');
var odayBtn=document.getElementById('oday'), surpriseBtn=document.getElementById('surprise');
var state={before:null,after:null,q:'',member:'',loading:false,done:false,searching:false,mode:'feed',regex:false};
var lastDay='';
var THREAD=null;
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmt(ts){var d=new Date(ts);return d.toLocaleString(undefined,{month:'short',day:'numeric',year:'numeric',hour:'2-digit',minute:'2-digit'});}
function dayKey(ts){var d=new Date(ts);return d.getFullYear()+'-'+(d.getMonth()+1)+'-'+d.getDate();}
function dayLabel(ts){var d=new Date(ts),t=new Date();var key=dayKey(ts);
var today=dayKey(t.getTime());var yest=dayKey(t.getTime()-864e5);
if(key===today)return 'Today';if(key===yest)return 'Yesterday';return d.toLocaleDateString(undefined,{weekday:'long',month:'long',day:'numeric',year:'numeric'});}
function yearsAgo(ts){var d=new Date(ts),t=new Date();if(d.getMonth()!==t.getMonth()||d.getDate()!==t.getDate())return null;var y=t.getFullYear()-d.getFullYear();return y>0?y:null;}
function sentBg(s){if(s==null)return '';if(s>0.15)return ' sent-pos';if(s<-0.15)return ' sent-neg';return '';}
function mediaHTML(m){
var out='';
if(m.photo_uris&&m.photo_uris.length){m.photo_uris.slice(0,3).forEach(function(u){out+='<div class="media"><img loading="lazy" src="/t/'+SLUG+'/media/'+esc(u)+'" alt=""/></div>';});}
if(m.has_gif&&m.gif_uris.length){m.gif_uris.slice(0,2).forEach(function(u){out+='<div class="media"><img loading="lazy" src="/t/'+SLUG+'/media/'+esc(u)+'" alt=""/></div>';});}
if(m.has_video&&m.video_uris.length){m.video_uris.slice(0,2).forEach(function(u){out+='<div class="media"><video controls preload="metadata" src="/t/'+SLUG+'/media/'+esc(u)+'"></video></div>';});}
if(m.has_audio&&m.audio_uris.length){m.audio_uris.slice(0,2).forEach(function(u){out+='<div class="media"><audio controls preload="metadata" src="/t/'+SLUG+'/media/'+esc(u)+'"></audio></div>';});}
if(m.has_file&&m.file_uris.length){m.file_uris.slice(0,3).forEach(function(u){out+='<div class="media"><a href="/t/'+SLUG+'/media/'+esc(u)+'" target="_blank">'+esc(u.split('/').pop())+'</a></div>';});}
if(m.has_file&&!m.file_uris.length){m.file_names.slice(0,3).forEach(function(n){out+='<div class="media">[file: '+esc(n)+']</div>';});}
if(m.has_sticker)out+='<div class="media sticker">[sticker]</div>';
if(m.link)out+='<div class="link"><a href="'+esc(m.link)+'" target="_blank" rel="noopener noreferrer">'+esc(m.link)+'</a></div>';
if(m.mtype==='Call')out+='<div class="call">[call]</div>';
if(m.is_unsent)out+='<span class="unsent">(unsent)</span>';
if(m.is_taken_down)out+='<span class="unsent">(removed)</span>';
return out;}
function msgHTML(m){
var ya=yearsAgo(m.ts);
var quote=m.reply_to?'<div class="quote">'+esc(m.reply_to.sender)+': '+esc(m.reply_to.snippet)+'</div>':'';
var reactions=m.reactions.map(function(r){return '<span class="ra">'+esc(r.reaction)+' '+esc(r.actor)+'</span>';}).join('');
var badge=ya?'<span class="badge">'+ya+'y ago</span>':'';
var body='<div class="text">'+esc(m.content)+'</div>';
var mhtml=mediaHTML(m);
return '<div class="msg'+sentBg(m.sentiment)+'"><div class="dot" style="background:'+esc(m.color)+'">'+esc(m.sender[0]).toUpperCase()+'</div>'+
'<div class="body"><div class="sender">'+esc(m.sender)+'<span class="time">'+fmt(m.ts)+'</span>'+badge+'</div>'+
quote+body+mhtml+(reactions?'<div class="reactions">'+reactions+'</div>':'')+'</div></div>';}
function appendMsgs(msgs){msgs.forEach(function(m){var k=dayKey(m.ts);if(k!==lastDay){if(lastDay!=='')feed.appendChild(document.createElement('div'));var d=document.createElement('div');d.className='day';d.textContent=dayLabel(m.ts);feed.appendChild(d);lastDay=k;}feed.insertAdjacentHTML('beforeend',msgHTML(m));});}
function apiURL(){var u='/t/'+SLUG+'/api/messages?limit=400';
if(state.q){u+='&q='+encodeURIComponent(state.q);}
if(state.regex)u+='&re=1';
if(state.member)u+='&member='+encodeURIComponent(state.member);
if(state.before!=null)u+='&before='+state.before;
if(state.after!=null)u+='&after='+state.after;
return u;}
function loadMore(){if(state.loading||state.done)return;state.loading=true;loader.textContent='Loading...';
fetch(apiURL()).then(function(r){return r.json();}).then(function(data){
state.loading=false;
if(state.searching){feed.innerHTML='';lastDay='';
countEl.textContent=data.truncated?('showing '+data.shown+' of '+data.total_matches+' match(es)'):(data.total_matches+' match(es)');
appendMsgs(data.messages);if(data.total_matches<1){feed.innerHTML='<div class="empty">No matches.</div>';}state.done=true;loader.style.display='none';return;}
appendMsgs(data.messages);
if(data.next_before!=null){state.before=data.next_before;}else if(data.next_after!=null){state.after=data.next_after;}else{state.done=true;}
countEl.textContent=feed.querySelectorAll('.msg').length+' / '+THREAD.total.toLocaleString()+' messages';
if(state.done){loader.textContent='End of history.';}}).catch(function(){state.loading=false;loader.textContent='Error loading.';});}
function reset(mode){state.loading=false;state.done=false;state.before=null;state.after=null;state.mode='feed';feed.innerHTML='';lastDay='';loader.style.display='';if(mode==='search'){state.searching=true;}else{state.searching=false;
/* Oldest-first walks forward from the start of history; leaving both cursors
   null always paginates backwards from the newest message. */
if(orderEl.value==='oldest')state.after=0;}
loadMore();}
function onThisDay(dateStr){state.mode='day';state.done=false;state.loading=false;feed.innerHTML='';lastDay='';loader.style.display='';
fetch('/t/'+SLUG+'/api/day?date='+encodeURIComponent(dateStr)).then(function(r){return r.json();}).then(function(data){
loader.style.display='none';state.done=true;
var byYear={};data.messages.forEach(function(m){var y=new Date(m.ts).getFullYear();(byYear[y]=byYear[y]||[]).push(m);});
var years=Object.keys(byYear).sort();
countEl.textContent=data.total+' message(s) on '+dateStr.slice(5)+' across '+(years.length||0)+' year(s)';
if(years.length===0){feed.innerHTML='<div class="empty">Nothing happened on this day.</div>';return;}
years.forEach(function(y){var d=document.createElement('div');d.className='day';d.textContent=y+' ('+byYear[y].length+' message'+(byYear[y].length===1?'':'s')+')';feed.appendChild(d);byYear[y].forEach(function(m){feed.insertAdjacentHTML('beforeend',msgHTML(m));});});
}).catch(function(){loader.style.display='none';state.done=true;feed.innerHTML='<div class="empty">Error loading.</div>';});}
function randomMemory(){state.mode='random';state.done=true;state.loading=false;feed.innerHTML='';lastDay='';loader.style.display='none';
fetch('/t/'+SLUG+'/api/random').then(function(r){return r.json();}).then(function(d){
countEl.textContent='Random memory (click again for another)';
feed.insertAdjacentHTML('beforeend',msgHTML(d.message));
feed.insertAdjacentHTML('beforeend','<div style="text-align:center;margin-top:16px"><button onclick="randomMemory()">Another</button></div>');
}).catch(function(){feed.innerHTML='<div class="empty">Error loading.</div>';});}
var io=new IntersectionObserver(function(entries){if(entries[0].isIntersecting)loadMore();},{rootMargin:'600px'});
io.observe(loader);
qEl.addEventListener('input',function(){var v=qEl.value.trim();if(state.q===v)return;state.q=v;reset(v?'search':'');});
memberEl.addEventListener('change',function(){state.member=memberEl.value;reset(state.q?'search':'');});
orderEl.addEventListener('change',function(){reset(state.q?'search':'');});
jumpEl.addEventListener('change',function(){if(!jumpEl.value)return;var d=new Date(jumpEl.value+'T00:00:00');
if(orderEl.value==='newest'){state.before=d.getTime()+864e5;state.after=null;}else{state.after=d.getTime()-864e5;state.before=null;}
state.mode='feed';state.done=false;feed.innerHTML='';lastDay='';loadMore();});
reBtn.addEventListener('click',function(){state.regex=!state.regex;reBtn.classList.toggle('active',state.regex);
if(state.q)reset('search');});
odayBtn.addEventListener('click',function(){onThisDay(jumpEl.value||new Date().toISOString().slice(0,10));});
surpriseBtn.addEventListener('click',randomMemory);
themeBtn.addEventListener('click',function(){var t=document.documentElement.getAttribute('data-theme')==='light'?'dark':'light';applyTheme(t);localStorage.setItem('cf-theme',t);});
function applyTheme(t){if(t==='light'){document.documentElement.setAttribute('data-theme','light');themeBtn.textContent='Dark';}else{document.documentElement.removeAttribute('data-theme');themeBtn.textContent='Light';}}
if(localStorage.getItem('cf-theme')==='light')applyTheme('light');
fetch('/t/'+SLUG+'/api/thread').then(function(r){return r.json();}).then(function(t){
THREAD=t;
memberEl.innerHTML='<option value="">Everyone</option>'+t.members.map(function(m){return '<option value="'+esc(m.name)+'">'+esc(m.name)+' ('+m.count.toLocaleString()+')</option>';}).join('');
document.title=t.title+' | chat-flashback';
loadMore();});
</script></body></html>"""


def run_server(threads, port, output_dir):
    indexed = {t["slug"]: ThreadIndex(t["slug"], t["title"], t["thread_dir"], t["msgs"])
               for t in threads}
    handler = make_handler(indexed, Path(output_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    for t in indexed.values():
        print(f"  {t.title}: {len(t.msgs):,} messages  ->  "
              f"http://127.0.0.1:{port}/t/{t.slug}/")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping.")
        server.shutdown()
