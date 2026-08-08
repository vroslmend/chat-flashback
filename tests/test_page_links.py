"""Every link on every generated page has to resolve.

The bug this guards against: the analyzer wrote eight pages and linked all of
them from the report's header, but the server routed three. Seven of the eight
links answered 404 while being served, and nothing caught it because no test
ever followed a link.

So these tests generate a real report, serve it, and walk it.
"""
import re
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import analyze_chat as ac
import chatpage
from chat_ui import run_server
from server_utils import wait_for_server

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"

# href="..." and href='...', since the generator uses both.
_HREF = re.compile(r"""href=["']([^"']+)["']""")
_SCRIPT = re.compile(r"<script\b.*?</script>", re.S)


def _links(html):
    """The hrefs in the markup.

    Scripts are stripped first: the reader builds media URLs by concatenation,
    and `href="/t/' + SLUG` reads as a link to `/t/` if you scan the source
    rather than the document.
    """
    return {h for h in _HREF.findall(_SCRIPT.sub("", html)) if _followable(h)}


def _generate(tmp_path):
    """The full pipeline over the sample export."""
    out = tmp_path / "out"
    assert ac.main(["--input", str(SAMPLE), "--output", str(out)]) == 0
    return out, ac._slug("Saturday Squad")


def _serve(out_dir, slug):
    threads = [{"slug": slug, "title": "Saturday Squad", "thread_dir": str(SAMPLE),
                "msgs": ac.normalize_messages(_raw_messages())}]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    threading.Thread(target=run_server, args=(threads, port, out_dir),
                     kwargs={"build_index": False}, daemon=True).start()
    wait_for_server(port)
    return port


def _raw_messages():
    import json
    msgs = []
    for i in range(1, 5):
        with open(SAMPLE / f"message_{i}.json", encoding="utf-8") as fh:
            msgs.extend(json.load(fh)["messages"])
    return msgs


def _status(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _followable(href):
    """Links that address another page of ours, as opposed to a place on this
    one or somewhere else entirely."""
    if not href or href.startswith(("#", "http://", "https://", "mailto:", "data:")):
        return False
    return True


def test_every_link_on_every_generated_page_resolves_while_served(tmp_path):
    out, slug = _generate(tmp_path)
    port = _serve(out, slug)
    base = f"http://127.0.0.1:{port}/t/{slug}/"

    pages = sorted(p.name for p in (out / slug).glob("*.html"))
    assert len(pages) >= 10, pages

    broken = []
    for page in pages:
        html = (out / slug / page).read_text(encoding="utf-8")
        for href in sorted(_links(html)):
            # Resolved against the page it sits on, the way a browser does.
            code = _status(urljoin(base + page, href))
            if code != 200:
                broken.append(f"{page} -> {href} ({code})")
    assert not broken, "dead links while served:\n" + "\n".join(broken)


def test_the_reader_itself_links_only_to_pages_that_exist(tmp_path):
    out, slug = _generate(tmp_path)
    port = _serve(out, slug)
    base = f"http://127.0.0.1:{port}/t/{slug}/"
    with urllib.request.urlopen(base) as r:
        reader = r.read().decode("utf-8")

    hrefs = sorted(_links(reader))
    assert hrefs, "the reader offers no way out of itself"
    broken = [h for h in hrefs if _status(urljoin(base, h)) != 200]
    assert not broken, broken


def test_nav_is_the_same_set_on_every_page(tmp_path):
    """One builder, one list. The old code filtered sibling links on the report
    and not on the narrative pages, so they disagreed about what existed."""
    out, slug = _generate(tmp_path)
    seen = {}
    for page in (out / slug).glob("*.html"):
        html = page.read_text(encoding="utf-8")
        nav = re.search(r'<nav class="cf-nav"[^>]*>(.*?)</nav>', html, re.S)
        assert nav, f"{page.name} has no shared nav"
        seen[page.name] = tuple(sorted(_HREF.findall(nav.group(1))))
    assert len(set(seen.values())) == 1, seen


def test_current_page_is_marked_not_dropped(tmp_path):
    """Links used to be removed from the nav on the page they pointed at, so
    every item shifted position as you moved around."""
    out, slug = _generate(tmp_path)
    for name in ("report.html", "eras.html", "quiz.html", "year_in_review.html"):
        html = (out / slug / name).read_text(encoding="utf-8")
        nav = re.search(r'<nav class="cf-nav"[^>]*>(.*?)</nav>', html, re.S).group(1)
        assert f'href="{name}"' in nav, f"{name} dropped itself from the nav"
        assert nav.count('aria-current="page"') == 1, name


def test_member_and_year_pages_are_reachable_from_the_nav(tmp_path):
    """They are not nav entries themselves, so they hang off an index."""
    out, slug = _generate(tmp_path)
    index = (out / slug / "members.html").read_text(encoding="utf-8")
    for page in (out / slug).glob("member_*.html"):
        assert page.name in index, f"{page.name} is not on the members index"
    years = (out / slug / "year_in_review.html").read_text(encoding="utf-8")
    for page in (out / slug).glob("year_2*.html"):
        assert page.name in years, f"{page.name} is not on the year index"


def test_pages_carry_one_theme_mechanism(tmp_path):
    """Three stylesheets meant three theme toggles, two of which forgot the
    choice on every navigation."""
    out, slug = _generate(tmp_path)
    for page in (out / slug).glob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert "cf-theme" in html, page.name
        # The old per-page toggle wrote to body.dataset and persisted nothing.
        assert "document.body.dataset.theme" not in html, page.name
        assert 'id="cf-theme"' in html, page.name


def test_reader_link_appears_only_when_a_reader_is_running(tmp_path):
    """On disk there is nothing to link to, so the hook stays a comment."""
    out, slug = _generate(tmp_path)
    on_disk = (out / slug / "report.html").read_text(encoding="utf-8")
    hook = chatpage.reader_hook_script("./")
    # The shared script always *tests* window.CF_READER; only the server ever
    # assigns it, which is what makes the reader link and the jump links appear.
    assert chatpage.READER_HOOK in on_disk
    assert hook not in on_disk

    port = _serve(out, slug)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/t/{slug}/report.html") as r:
        served = r.read().decode("utf-8")
    assert chatpage.READER_HOOK not in served
    assert hook in served


def test_quoted_messages_carry_a_timestamp_to_jump_to(tmp_path):
    out, slug = _generate(tmp_path)
    sessions = (out / slug / "sessions.html").read_text(encoding="utf-8")
    stamps = re.findall(r"data-ts='(\d+)'", sessions)
    assert len(stamps) >= 5, "conversations quote messages but offer no way in"
    assert all(len(s) == 13 for s in stamps), stamps[:3]


def test_traversal_out_of_the_output_directory_is_refused(tmp_path):
    out, slug = _generate(tmp_path)
    port = _serve(out, slug)
    base = f"http://127.0.0.1:{port}/t/{slug}/"
    for attempt in ("../../analyze_chat.py", "..%2f..%2fanalyze_chat.py",
                    "....//report.html"):
        assert _status(base + attempt) == 404, attempt
