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


def write_export(tmp_path, messages, title=THREAD, participants=None):
    """A minimal Messenger export folder holding the given raw messages."""
    thread = tmp_path / "export" / f"{title}_1"
    thread.mkdir(parents=True, exist_ok=True)
    names = participants or sorted({m["sender_name"] for m in messages})
    (thread / "message_1.json").write_text(
        json.dumps({"title": title,
                    "participants": [{"name": n} for n in names],
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


def generate(tmp_path, messages, *extra, participants=None):
    thread = write_export(tmp_path, messages, participants=participants)
    out = tmp_path / "out"
    code = ac.main(["--input", str(thread), "--output", str(out), *extra])
    assert code == 0
    return out / ac._slug(THREAD)


def topics_of(out_dir):
    text = (out_dir / "summary.md").read_text(encoding="utf-8")
    return text.split("## What the chat was about")[1].split("##")[0]


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
    topics = topics_of(generate(tmp_path, msgs))
    assert "cricket" in topics
    assert "alice" not in topics
    assert "bob" not in topics


def test_a_listed_participant_who_never_posted_is_not_a_topic(tmp_path):
    """Someone can be in the chat and send nothing -- a deleted account still
    gets talked about by name."""
    msgs = [raw("Alice" if i % 2 else "Bob", BASE + timedelta(minutes=i),
                "ahsan cricket") for i in range(12)]
    topics = topics_of(generate(tmp_path, msgs,
                                participants=["Alice", "Bob", "Ahsan Raza"]))
    assert "cricket" in topics
    assert "ahsan" not in topics


def test_names_flag_covers_people_the_export_never_lists(tmp_path):
    """The wiped account shows up as "Facebook user", so its real name is
    nowhere in the export and has to be supplied."""
    msgs = [raw("Alice" if i % 2 else "Bob", BASE + timedelta(minutes=i),
                "ahsan cricket") for i in range(12)]
    topics = topics_of(generate(tmp_path, msgs, "--names", "Ahsan, Shahood"))
    assert "cricket" in topics
    assert "ahsan" not in topics


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


# --------------------------------------------------------------------------- #
# narrative pages                                                              #
# --------------------------------------------------------------------------- #

def test_the_report_links_every_narrative_page_it_wrote(tmp_path):
    out = generate(tmp_path, chatter())
    report = (out / "report.html").read_text(encoding="utf-8")
    for page in ["group_history.html", "relationships.html", "eras.html"]:
        assert (out / page).is_file()
        assert f'href="{page}"' in report


def test_a_page_is_written_for_every_member(tmp_path):
    out = generate(tmp_path, chatter())
    report = (out / "report.html").read_text(encoding="utf-8")
    for member in ("alice", "bob"):
        assert (out / f"member_{member}.html").is_file()
        assert f"member_{member}.html" in report


def test_the_member_page_reports_their_own_years(tmp_path):
    msgs = chatter() + [raw("Alice", BASE.replace(year=2021), "cricket again next year")]
    out = generate(tmp_path, msgs)
    page = (out / "member_alice.html").read_text(encoding="utf-8")
    assert "2020" in page and "2021" in page
    assert "Active 2020-01-01 to 2021-01-01" in page


def test_the_group_history_page_reads_back_messenger_events(tmp_path):
    msgs = chatter() + [
        raw("Alice", BASE + timedelta(days=1), "Alice named the group squad goals."),
        raw("Alice", BASE + timedelta(days=2),
            "Alice set the nickname for Bob to speed racer."),
    ]
    page = (generate(tmp_path, msgs) / "group_history.html").read_text(encoding="utf-8")
    assert "squad goals" in page
    assert "speed racer" in page
    assert "Currently called" in page


def test_group_events_stay_out_of_the_vocabulary_they_are_reported_from(tmp_path):
    """The page reads them; the word counts still must not."""
    msgs = chatter() + [raw("Alice", BASE + timedelta(days=1),
                            "Alice named the group squad goals.")]
    out = generate(tmp_path, msgs)
    assert "squad goals" in (out / "group_history.html").read_text(encoding="utf-8")
    assert "squad" not in topics_of(out)


def test_the_eras_page_explains_its_own_rule(tmp_path):
    page = (generate(tmp_path, chatter()) / "eras.html").read_text(encoding="utf-8")
    assert "opens a new era" in page
    assert "born" in page.lower()


def test_narratives_can_be_skipped(tmp_path):
    out = generate(tmp_path, chatter(), "--skip", "narratives")
    assert not (out / "eras.html").exists()
    assert not (out / "member_alice.html").exists()
    assert 'href="eras.html"' not in (out / "report.html").read_text(encoding="utf-8")


def test_summary_json_carries_the_narratives(tmp_path):
    msgs = chatter() + [
        raw("Alice", BASE + timedelta(days=1), "Alice named the group squad goals."),
    ]
    out = generate(tmp_path, msgs, "--json")
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["group_history"]["current_name"] == "squad goals"
    assert payload["relationships"]["pairs"]
    assert "Alice" in payload["members"]
    assert "top_reacted" not in payload["members"]["Alice"]


def test_the_trendsetters_page_and_its_report_section_name_the_same_member(tmp_path):
    """Alice says "yeeted" first, three others pick it up, and it sticks around
    long enough to clear the band."""
    msgs = [raw("Alice", BASE + timedelta(hours=i), "ordinary talk here")
            for i in range(110)]
    msgs += [raw("Bob", BASE + timedelta(hours=i, minutes=1), "ordinary talk here")
             for i in range(30)]
    start = BASE + timedelta(days=120)
    msgs.append(raw("Alice", start, "yeeted the whole thing"))
    msgs += [raw(who, start + timedelta(days=n), "yeeted again")
             for n, who in enumerate(["Bob", "Dana", "Charlie"], start=1)]
    msgs += [raw("Alice", start + timedelta(days=5, hours=h), "yeeted yeeted")
             for h in range(10)]

    out = generate(tmp_path, msgs, "--json")
    page = (out / "trendsetters.html").read_text(encoding="utf-8")
    assert "yeeted" in page
    assert "3 others" in page
    report = (out / "report.html").read_text(encoding="utf-8")
    assert '<section id="trendsetters">' in report
    assert 'href="trendsetters.html"' in report
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["trendsetters"]["members"][0]["member"] == "Alice"
    assert payload["trendsetters"]["words"][0]["word"] == "yeeted"


def test_extra_stopwords_reach_the_narrative_pages(tmp_path):
    """Run as a script, analyze_chat is `__main__`, and chatstats importing it
    by name used to get a second copy whose STOPWORDS never saw
    --stopwords-file. Member pages then listed the very words the user asked to
    drop. Only a subprocess reproduces it: under pytest the module is imported
    normally and there is only ever one copy."""
    import os
    import subprocess
    import sys as _sys
    from pathlib import Path

    msgs = [raw("Alice" if i % 2 else "Bob", BASE + timedelta(minutes=i),
                "cricket practice again") for i in range(30)]
    thread = write_export(tmp_path, msgs)
    stops = tmp_path / "stops.txt"
    stops.write_text("cricket\n", encoding="utf-8")
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [_sys.executable, "analyze_chat.py", "--input", str(thread),
         "--output", str(out), "--stopwords-file", str(stops),
         "--skip", "jokes,sentiment,wordcloud"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(ac.__file__).resolve().parent))
    assert result.returncode == 0, result.stderr
    page = (out / ac._slug(THREAD) / "member_alice.html").read_text(encoding="utf-8")
    words = page.split("Words of that year")[1]
    assert "practice" in words
    assert "cricket" not in words
