"""Tests for what the generated reports actually say.

These drive a real export through `main()` rather than poking at the analysis
functions, because every bug here was in the rendering: numbers that were right
in `analyses` and wrong on the page.
"""
import json
from collections import Counter
from datetime import datetime, timedelta

import analyze_chat as ac
from test_correctness import BASE, mk

THREAD = "test_thread"


def write_export(tmp_path, messages, title=THREAD):
    """A minimal Messenger export folder holding the given raw messages."""
    thread = tmp_path / "export" / f"{title}_1"
    thread.mkdir(parents=True, exist_ok=True)
    (thread / "message_1.json").write_text(
        json.dumps({"title": title,
                    "participants": [{"name": n} for n in
                                     sorted({m["sender_name"] for m in messages})],
                    "messages": messages}),
        encoding="utf-8")
    return thread


def raw(sender, dt, content, **kw):
    m = {"sender_name": sender, "timestamp_ms": int(dt.timestamp() * 1000),
         "content": content}
    m.update(kw)
    return m


def chatter(n=40, start=BASE):
    """Two people talking, enough to fill every section of the report."""
    out = []
    for i in range(n):
        who = "Alice" if i % 2 else "Bob"
        out.append(raw(who, start + timedelta(minutes=3 * i),
                       f"message number {i} about cricket?"))
    # A swear and a shared link, so the sections that need them are exercised.
    out.append(raw("Alice", start + timedelta(minutes=3 * n), "damn that was close"))
    out.append(raw("Bob", start + timedelta(minutes=3 * n + 2), "look at this",
                   share={"link": "https://www.youtube.com/watch?v=abc"}))
    return out


def generate(tmp_path, messages, *extra):
    thread = write_export(tmp_path, messages)
    out = tmp_path / "out"
    code = ac.main(["--input", str(thread), "--output", str(out), *extra])
    assert code == 0
    return out / ac._slug(THREAD)


# --------------------------------------------------------------------------- #
# response speed                                                               #
# --------------------------------------------------------------------------- #

def test_response_speed_table_shows_seconds_not_a_row_of_ties(tmp_path):
    msgs = [raw("Alice", BASE, "hey"),
            raw("Bob", BASE + timedelta(seconds=6), "yo"),
            raw("Alice", BASE + timedelta(seconds=20), "sup"),
            raw("Bob", BASE + timedelta(seconds=26), "nm")]
    text = (generate(tmp_path, msgs) / "summary.md").read_text(encoding="utf-8")
    speed = text.split("## Response speed")[1].split("##")[0]
    assert "6 s" in speed
    assert "0.1 min" not in speed


def test_a_chart_left_by_an_earlier_run_is_not_embedded(tmp_path):
    """Charts were globbed off disk, so a stale png from a run with different
    flags rode along in the new report as if it belonged to it."""
    out = generate(tmp_path, chatter())
    stale = out / "wordcloud_someone_who_left.png"
    stale.write_bytes((out / "wordcloud.png").read_bytes())
    out = generate(tmp_path, chatter())
    assert "wordcloud_someone_who_left.png" not in \
        (out / "report.html").read_text(encoding="utf-8")
    assert "wordcloud_someone_who_left.png" not in \
        (out / "summary.md").read_text(encoding="utf-8")


def test_a_chat_with_no_vocabulary_does_not_crash_the_run(tmp_path):
    """WordCloud raises on an empty frequency dict, taking the whole report down."""
    msgs = [raw("Alice", BASE + timedelta(minutes=i), "ok yeah") for i in range(3)]
    msgs += [raw("Bob", BASE + timedelta(minutes=i, seconds=30), "yes ok")
             for i in range(3)]
    out = generate(tmp_path, msgs)
    assert (out / "summary.md").exists()
    assert not (out / "wordcloud.png").exists()


def test_fastest_replier_headline_uses_a_unit_that_shows_the_gap():
    speed = {"table": [{"member": "Alice", "median_s": 6, "median_m": 0.1}]}
    line = [l for l in ac.insights({"member_msgs": Counter({"Alice": 2}), "total": 2},
                                   {"speed": speed}) if "fastest" in l][0]
    assert "6 s" in line
    assert "0.1 min" not in line


def test_question_speed_chart_is_drawn_in_seconds(tmp_path, monkeypatch):
    drawn = {}
    real_bar = ac._bar

    def spy(fig, ax, labels, values, title, colors=None):
        drawn[title] = list(values)
        return real_bar(fig, ax, labels, values, title, colors)

    monkeypatch.setattr(ac, "_bar", spy)
    msgs = [raw("Alice", BASE, "where?"),
            raw("Bob", BASE + timedelta(seconds=8), "there"),
            raw("Alice", BASE + timedelta(seconds=30), "when?"),
            raw("Bob", BASE + timedelta(seconds=38), "now")]
    generate(tmp_path, msgs)
    title = [t for t in drawn if t.startswith("Median time to answer")][0]
    assert "second" in title
    assert drawn[title] == [8]


# --------------------------------------------------------------------------- #
# html parity                                                                  #
# --------------------------------------------------------------------------- #

def test_report_html_carries_the_tables_the_summary_has(tmp_path):
    html = (generate(tmp_path, chatter()) / "report.html").read_text(encoding="utf-8")
    for section in ["personalities", "speed", "swear", "starters", "ghosting",
                    "hourly", "lengths", "domains"]:
        assert f'<section id="{section}">' in html, f"missing section {section}"


def test_report_html_nav_reaches_every_section(tmp_path):
    html = (generate(tmp_path, chatter()) / "report.html").read_text(encoding="utf-8")
    import re
    ids = set(re.findall(r'<section id="([^"]+)">', html))
    linked = set(re.findall(r'<a href="#([^"]+)">', html))
    assert ids == linked


def test_summary_json_keeps_both_the_media_total_and_the_breakdown(tmp_path):
    """Two payload entries were both called "media", so the total was lost."""
    msgs = chatter() + [raw("Alice", BASE + timedelta(hours=7), None,
                            photos=[{"uri": "photos/a.jpg"}])]
    out = generate(tmp_path, msgs, "--json")
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert data["media"] == 1
    assert data["media_by_member"]["Alice"]["photos"] == 1


def test_topic_words_are_not_just_everyones_names(tmp_path):
    """Members address each other constantly, so names swamped the topics."""
    msgs = [raw("Alice" if i % 2 else "Bob", BASE + timedelta(minutes=i),
                "alice bob cricket")
            for i in range(12)]
    text = (generate(tmp_path, msgs) / "summary.md").read_text(encoding="utf-8")
    topics = text.split("## What the chat was about")[1].split("##")[0]
    assert "cricket" in topics
    assert "alice" not in topics
    assert "bob" not in topics


# --------------------------------------------------------------------------- #
# bots                                                                         #
# --------------------------------------------------------------------------- #

def test_bot_is_labelled_in_the_leaderboard(tmp_path):
    msgs = chatter() + [raw("Meta AI", BASE + timedelta(hours=5), "I am here to help")]
    text = (generate(tmp_path, msgs) / "summary.md").read_text(encoding="utf-8")
    assert "Meta AI (bot)" in text


def test_best_vibes_headline_skips_bots():
    stats = {"member_msgs": Counter({"Alice": 5}), "total": 5}
    analyses = {"sentiment": {"per_member": {"Meta AI": 0.9, "Alice": 0.1},
                              "per_year": {}, "messages_scored": 5}}
    assert any("Best vibes come from Alice" in line
               for line in ac.insights(stats, analyses))


# --------------------------------------------------------------------------- #
# copy-paste floods                                                            #
# --------------------------------------------------------------------------- #

def test_floods_are_reported_rather_than_silently_capped(tmp_path):
    msgs = chatter() + [raw("Bob", BASE + timedelta(hours=6), "optix " * 50)]
    text = (generate(tmp_path, msgs) / "summary.md").read_text(encoding="utf-8")
    assert "Copy-paste floods" in text
    assert "**Copy-paste floods**: 1" in text


# --------------------------------------------------------------------------- #
# missing export features                                                      #
# --------------------------------------------------------------------------- #

def test_reply_chains_say_why_they_are_missing(tmp_path):
    """No message carries an id, so chains cannot be rebuilt -- say so."""
    text = (generate(tmp_path, chatter()) / "summary.md").read_text(encoding="utf-8")
    chains = text.split("## Reply chains")[1].split("##")[0]
    assert "unavailable" in chains.lower()


# --------------------------------------------------------------------------- #
# formatting slips                                                             #
# --------------------------------------------------------------------------- #

def test_longest_session_shows_the_end_date_too(tmp_path):
    """A session crossing midnight read '10:25 - 01:22' and looked backwards."""
    msgs = [raw("Alice", BASE.replace(hour=23, minute=40), "late one"),
            raw("Bob", BASE.replace(hour=23, minute=50), "yeah"),
            raw("Alice", BASE.replace(hour=23, minute=59), "night"),
            raw("Bob", (BASE + timedelta(days=1)).replace(hour=0, minute=10), "ok")]
    text = (generate(tmp_path, msgs) / "summary.md").read_text(encoding="utf-8")
    session = text.split("Longest single session")[1].split("\n")[0]
    assert session.count("-") >= 2 and "2020-01-02 00:10" in session


def test_console_and_summary_agree_on_the_answered_percentage(tmp_path, capsys):
    """The console floored the percentage while the summary rounded it: 98 vs 99."""
    out = generate(tmp_path, chatter())
    printed = capsys.readouterr().out
    text = (out / "summary.md").read_text(encoding="utf-8")
    console_pct = [l for l in printed.splitlines()
                   if "Questions:" in l][0].split(",")[1].split("%")[0].strip()
    summary_pct = text.split("got a reply within an hour (")[1].split("%")[0]
    assert console_pct == summary_pct


# --------------------------------------------------------------------------- #
# --top                                                                        #
# --------------------------------------------------------------------------- #

def test_top_flag_controls_the_word_and_emoji_charts(tmp_path, monkeypatch):
    drawn = {}
    real_bar = ac._bar

    def spy(fig, ax, labels, values, title, colors=None):
        drawn[title] = list(labels)
        return real_bar(fig, ax, labels, values, title, colors)

    monkeypatch.setattr(ac, "_bar", spy)
    faces = "\U0001f600\U0001f601\U0001f602\U0001f603\U0001f604\U0001f605"
    msgs = [raw("Alice", BASE + timedelta(minutes=i), f"word{i} {faces[i % 6]}")
            for i in range(40)]
    generate(tmp_path, msgs, "--top", "3")
    assert len(drawn["Top words"]) == 3
    assert len(drawn["Top emojis"]) == 3
