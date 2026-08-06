"""Numeric regression tests.

Each test here pins down a statistic that was silently wrong at some point.
The rest of the suite checks that files get written and the pipeline exits 0,
which every one of these bugs passed straight through.
"""
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")
BASE = datetime(2020, 1, 1, 12, 0)


def mk(sender, dt, content, **kw):
    """A normalized message, as normalize_messages would produce it."""
    m = {"id": None, "sender": sender, "ts_ms": int(dt.timestamp() * 1000), "dt": dt,
         "content": content, "mtype": "Generic", "reactions": [], "has_photo": False,
         "has_sticker": False, "has_gif": False, "has_video": False, "has_audio": False,
         "has_file": False, "has_media": False, "photo_uris": [], "gif_uris": [],
         "video_uris": [], "audio_uris": [], "file_uris": [], "file_names": [],
         "poll_question": None, "link": None, "call_duration": None, "reply_to": None,
         "is_unsent": False, "is_taken_down": False}
    m.update(kw)
    return m


# --------------------------------------------------------------------------- #
# tokenizing                                                                   #
# --------------------------------------------------------------------------- #

def test_tokenize_keeps_digits_inside_words():
    assert ac.tokenize("covid19 was rough") == ["covid19", "was", "rough"]
    assert ac.tokenize("b2b") == ["b2b"]


def test_tokenize_drops_bare_numbers():
    assert ac.tokenize("2020 top 10 songs") == ["top", "songs"]


# --------------------------------------------------------------------------- #
# text decoding                                                                #
# --------------------------------------------------------------------------- #

def test_decode_repairs_mojibake_already_parsed_by_json():
    """The common real-export case: json.loads has already turned the escapes
    into characters, so there is no "\\u00" left to look for."""
    mangled = "\U0001f621".encode("utf-8").decode("latin1")
    assert mangled != "\U0001f621"
    assert ac.decode_messenger_text(mangled) == "\U0001f621"


def test_decode_still_repairs_literal_escape_form():
    assert ac.decode_messenger_text("\\u00f0\\u009f\\u0091\\u008d") == "\U0001f44d"


def test_decode_leaves_plain_and_real_unicode_alone():
    assert ac.decode_messenger_text("hello there") == "hello there"
    assert ac.decode_messenger_text("مرحبا") == "مرحبا"
    assert ac.decode_messenger_text(None) is None
    assert ac.decode_messenger_text("") == ""


def test_decode_leaves_latin1_text_that_is_not_utf8_alone():
    """'café' is valid Latin-1 but not valid UTF-8, so it must survive."""
    assert ac.decode_messenger_text("café") == "café"


def test_mangled_emoji_are_counted_after_normalization():
    raw = [{"sender_name": "Alice", "timestamp_ms": 1609459200000,
            "content": "\U0001f621".encode("utf-8").decode("latin1")},
           {"sender_name": "Bob", "timestamp_ms": 1609459200001, "content": "plain",
            "reactions": [{"reaction": "❤".encode("utf-8").decode("latin1"),
                           "actor": "Alice"}]}]
    msgs = ac.normalize_messages(raw)
    assert msgs[0]["content"] == "\U0001f621"
    assert ac.split_emojis(msgs[0]["content"]) == ["\U0001f621"]
    assert msgs[1]["reactions"] == [("Alice", "❤")]


# --------------------------------------------------------------------------- #
# topic words                                                                  #
# --------------------------------------------------------------------------- #

def test_topic_words_scores_are_never_negative():
    """idf built from message counts went negative for any common word."""
    msgs = [mk("Alice", BASE + timedelta(hours=i), "pizza tonight") for i in range(200)]
    msgs.append(mk("Bob", BASE + timedelta(hours=300), "xylophone"))
    topics = ac.topic_words(msgs, top=10)
    scores = [w["score"] for w in topics["by_year"][2020]]
    assert scores, "expected topic words"
    assert all(s >= 0 for s in scores), f"negative tf-idf scores: {scores}"


def test_topic_words_ranks_frequent_words_above_one_offs():
    msgs = [mk("Alice", BASE + timedelta(hours=i), "pizza tonight") for i in range(200)]
    msgs.append(mk("Bob", BASE + timedelta(hours=300), "xylophone"))
    words = [w["word"] for w in ac.topic_words(msgs, top=3)["by_year"][2020]]
    assert words[0] in ("pizza", "tonight"), f"got {words}"
    assert "xylophone" not in words[:2]


def test_topic_words_favours_year_specific_vocabulary():
    """A word used in one year only should outrank an equally frequent one
    used in every year."""
    msgs = []
    for n, year in enumerate((2020, 2021, 2022)):
        start = BASE.replace(year=year)
        for i in range(30):
            msgs.append(mk("Alice", start + timedelta(hours=i), "always chatting"))
    start = BASE.replace(year=2021)
    for i in range(30):
        msgs.append(mk("Bob", start + timedelta(hours=100 + i), "wedding wedding"))
    words = [w["word"] for w in ac.topic_words(msgs, top=3)["by_year"][2021]]
    assert words[0] == "wedding", f"expected the year-specific word first, got {words}"


# --------------------------------------------------------------------------- #
# question dynamics                                                            #
# --------------------------------------------------------------------------- #

def test_question_answered_despite_earlier_message_in_same_run():
    """Gaps shrink towards the reply, so an early out-of-window message must
    not stop the scan for the later in-window ones."""
    msgs = [
        mk("Alice", BASE, "just thinking out loud"),
        mk("Alice", BASE + timedelta(hours=5), "anyone up?"),
        mk("Bob", BASE + timedelta(hours=5, minutes=1), "yes!"),
    ]
    q = ac.question_stats(msgs)
    assert q["total_questions"] == 1
    assert q["total_answered"] == 1, "question one minute before the reply counted unanswered"


def test_question_beyond_the_window_stays_unanswered():
    msgs = [
        mk("Alice", BASE, "anyone up?"),
        mk("Bob", BASE + timedelta(hours=3), "sorry, was asleep"),
    ]
    q = ac.question_stats(msgs)
    assert q["total_questions"] == 1
    assert q["total_answered"] == 0


# --------------------------------------------------------------------------- #
# emoji totals                                                                 #
# --------------------------------------------------------------------------- #

def test_sum_counters_adds_repeated_keys():
    """A dict comprehension over Counters keeps only the last value per key."""
    total = ac._sum_counters([Counter({"a": 100}), Counter({"a": 1}), Counter({"b": 50})])
    assert total["a"] == 101
    assert total["b"] == 50


def test_insights_picks_the_overall_favourite_emoji():
    analyses = {"emojis": {"per_year": {2020: Counter({"\U0001f600": 100}),
                                        2021: Counter({"\U0001f600": 1}),
                                        2022: Counter({"\U0001f602": 50})}}}
    out = ac.insights({"member_msgs": Counter(), "total": 0}, analyses)
    line = [l for l in out if "favorite emoji" in l]
    assert line, out
    assert "\U0001f600" in line[0], f"picked the wrong emoji: {line[0]}"
    assert "101" in line[0]


# --------------------------------------------------------------------------- #
# anonymization                                                                #
# --------------------------------------------------------------------------- #

def test_anonymize_does_not_leak_a_longer_name():
    """'Ann' must not rewrite the first half of 'Ann Smith'."""
    msgs = ([mk("Ann", BASE + timedelta(minutes=i), "yo") for i in range(20)]
            + [mk("Ann Smith", BASE + timedelta(minutes=40 + i), "hi") for i in range(3)]
            + [mk("Ann", BASE + timedelta(minutes=60), "Ann Smith came over")])
    out = ac.apply_anonymization(msgs, ac.anonymize_map(msgs))
    assert "Smith" not in out[-1]["content"], out[-1]["content"]
    assert out[-1]["content"] == "Person B came over"


def test_anonymize_replaces_every_member_name_in_text():
    msgs = ([mk("Alice", BASE + timedelta(minutes=i), "hey") for i in range(5)]
            + [mk("Bob", BASE + timedelta(minutes=10), "Alice told Bob about it")])
    out = ac.apply_anonymization(msgs, ac.anonymize_map(msgs))
    assert "Alice" not in out[-1]["content"]
    assert "Bob" not in out[-1]["content"]


def test_anonymize_pipeline_leaves_no_real_names(tmp_path):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path), "--anonymize"]) == 0
    text = (tmp_path / "saturday_squad" / "report.html").read_text(encoding="utf-8")
    # Embedded chart data is base64, which contains arbitrary letter runs.
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", text)
    for name in ("Alice", "Bob", "Charlie", "Dana"):
        assert name not in text


# --------------------------------------------------------------------------- #
# response speed                                                               #
# --------------------------------------------------------------------------- #

def test_response_speed_lists_members_who_never_replied():
    msgs = [mk("Alice", BASE + timedelta(seconds=i * 10), "spam") for i in range(10)]
    msgs.append(mk("Bob", BASE + timedelta(seconds=95), "ok"))
    table = ac.response_speed(msgs)["table"]
    assert {r["member"] for r in table} == {"Alice", "Bob"}
    alice = next(r for r in table if r["member"] == "Alice")
    assert alice["replies"] == 0
    assert alice["median_s"] is None


def test_ghost_pct_counts_turns_not_messages():
    """Ten messages in a row are one turn; it was answered, so 0% ghosted."""
    msgs = [mk("Alice", BASE + timedelta(seconds=i * 10), "spam") for i in range(10)]
    msgs.append(mk("Bob", BASE + timedelta(seconds=95), "ok"))
    msgs.append(mk("Alice", BASE + timedelta(seconds=100), "thanks"))
    alice = next(r for r in ac.response_speed(msgs)["table"] if r["member"] == "Alice")
    assert alice["turns"] == 1
    assert alice["ghost_pct"] == 0.0


def test_ghost_pct_flags_an_unanswered_turn():
    msgs = [
        mk("Alice", BASE, "hello?"),
        mk("Bob", BASE + timedelta(hours=6), "sorry"),
        mk("Alice", BASE + timedelta(hours=6, seconds=30), "np"),
    ]
    alice = next(r for r in ac.response_speed(msgs)["table"] if r["member"] == "Alice")
    assert alice["turns"] == 1
    assert alice["ghost_pct"] == 100.0


def test_fastest_replier_skips_members_without_a_median():
    msgs = [mk("Alice", BASE + timedelta(seconds=i * 10), "spam") for i in range(3)]
    msgs.append(mk("Bob", BASE + timedelta(seconds=25), "ok"))
    speed = ac.response_speed(msgs)
    fastest = ac._fastest_replier(speed)
    assert fastest is not None and fastest["member"] == "Bob"


# --------------------------------------------------------------------------- #
# running jokes                                                                #
# --------------------------------------------------------------------------- #

def test_signature_word_is_the_most_used_not_the_last_alphabetically():
    """Words a member uses exclusively all tie at ratio 1.0, and the tiebreak
    used to be the word itself, so signatures were always w/y/z words."""
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "apple") for i in range(500)]
    msgs += [mk("Alice", BASE + timedelta(minutes=600 + i), "zebra") for i in range(3)]
    msgs += [mk("Bob", BASE + timedelta(minutes=900 + i), "hello there friend")
             for i in range(5)]
    assert ac.personalities(msgs)["Alice"]["signature"] == "apple"


def test_urls_do_not_become_vocabulary():
    msgs = [mk("Alice", BASE + timedelta(minutes=i),
               "look https://www.youtube.com/watch?v=abcd amazing") for i in range(10)]
    ac.add_derived_fields(msgs)
    toks = set(msgs[0]["tokens"])
    assert "youtube" not in toks and "https" not in toks and "watch" not in toks
    assert "look" in toks and "amazing" in toks


def test_messenger_boilerplate_is_not_treated_as_vocabulary():
    """Messenger fills content with "X sent an attachment." for some media."""
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "halo sent an attachment.")
            for i in range(10)]
    msgs += [mk("Bob", BASE + timedelta(minutes=100), "i sent an email about it")]
    ac.add_derived_fields(msgs)
    assert msgs[0]["tokens"] == ()
    # A real sentence that merely mentions sending must survive.
    assert "email" in msgs[-1]["tokens"]
    assert ac.core_stats(msgs)["words"]["attachment"] == 0


def test_reaction_notice_boilerplate_counts_as_neither_words_nor_emoji():
    """"<nickname> reacted <emoji> to your message" is written by Messenger, so
    the nickname is not a phrase anyone said and the emoji was not typed."""
    msgs = [mk("Alice", BASE + timedelta(minutes=i),
               "speed racer reacted \U0001f606 to your message ") for i in range(10)]
    msgs += [mk("Bob", BASE + timedelta(minutes=100), "everyone reacted well \U0001f600")]
    ac.add_derived_fields(msgs)
    assert msgs[0]["tokens"] == ()
    assert msgs[0]["emojis"] == ()
    # A real message that happens to use the word "reacted" is untouched.
    assert "reacted" in msgs[-1]["tokens"]
    assert msgs[-1]["emojis"] == ("\U0001f600",)
    stats = ac.core_stats(msgs)
    assert stats["words"]["racer"] == 0
    assert stats["emojis"]["\U0001f606"] == 0


def test_stopwords_file_extends_the_builtin_list(tmp_path):
    words = tmp_path / "extra.txt"
    words.write_text("# comment\nhai\nnahi\n\n", encoding="utf-8")
    out = tmp_path / "out"
    before = set(ac.STOPWORDS)
    try:
        assert ac.main(["--input", SAMPLE, "--output", str(out),
                        "--stopwords-file", str(words)]) == 0
        assert "hai" in ac.STOPWORDS and "nahi" in ac.STOPWORDS
    finally:
        ac.STOPWORDS.clear()
        ac.STOPWORDS.update(before)


def test_shipped_hinglish_stopwords_load():
    path = Path(__file__).resolve().parents[1] / "stopwords" / "hinglish.txt"
    assert path.exists(), "stopwords/hinglish.txt should ship with the repo"
    terms = ac.load_track_file(str(path))
    assert "hai" in terms and "nahi" in terms and "bhi" in terms
    assert not any(t.startswith("#") for t in terms)


def test_laughter_slang_is_not_counted_as_profanity():
    assert "lmao" not in ac.CENSOR_WORDS
    assert "omg" not in ac.CENSOR_WORDS
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "lmao omg") for i in range(10)]
    assert ac.swear_stats(msgs)["total_hits"] == 0


def test_calls_report_unavailable_when_the_export_has_no_types():
    raw = [{"sender_name": "Alice", "timestamp_ms": 1609459200000, "content": "hi"}]
    stats = ac.core_stats(ac.normalize_messages(raw))
    assert stats["types_available"] is False
    label = dict(ac.all_time_totals(stats, {}))["Calls"]
    assert "unavailable" in label

    typed = [{"sender_name": "Alice", "timestamp_ms": 1609459200000,
              "content": "hi", "type": "Generic"}]
    stats = ac.core_stats(ac.normalize_messages(typed))
    assert stats["types_available"] is True
    assert "unavailable" not in dict(ac.all_time_totals(stats, {}))["Calls"]


def test_inside_jokes_ignores_member_names():
    msgs = []
    n = 0
    for year in (2020, 2021):
        for sender in ("Ammar Hassan", "Syed Jafri"):
            for _ in range(4):
                dt = BASE.replace(year=year) + timedelta(days=n)
                msgs.append(mk(sender, dt, "ammar hassan said the thing"))
                n += 1
    ac.add_derived_fields(msgs)
    phrases = {j["phrase"] for j in ac.inside_jokes(msgs)["jokes"]}
    assert not any("ammar" in p or "hassan" in p for p in phrases), phrases


def test_inside_jokes_collapses_fragments_of_a_longer_phrase():
    msgs = []
    n = 0
    for year in (2020, 2021):
        for sender in ("Alice", "Bob"):
            for _ in range(4):
                dt = BASE.replace(year=year) + timedelta(days=n)
                msgs.append(mk(sender, dt, "pizza friday night party"))
                n += 1
    ac.add_derived_fields(msgs)
    phrases = [j["phrase"] for j in ac.inside_jokes(msgs)["jokes"]]
    assert "pizza friday night party" in phrases
    # Its fragments have the same count, so they are the same joke.
    assert "pizza friday" not in phrases
    assert "friday night party" not in phrases


def test_inside_jokes_still_finds_a_four_word_phrase():
    """Bigram pruning must not drop longer phrases that do qualify."""
    msgs = []
    n = 0
    for year in (2020, 2021):
        for sender in ("Alice", "Bob"):
            for _ in range(3):
                dt = BASE.replace(year=year) + timedelta(days=n)
                msgs.append(mk(sender, dt, "pizza friday night party"))
                n += 1
    jokes = ac.inside_jokes(msgs)
    phrases = {j["phrase"] for j in jokes["jokes"]}
    assert "pizza friday night party" in phrases, phrases


def test_inside_jokes_ignores_phrases_below_min_count():
    msgs = [mk("Alice", BASE + timedelta(days=i), "pizza friday night") for i in range(2)]
    msgs += [mk("Bob", BASE.replace(year=2021) + timedelta(days=i), "totally different words")
             for i in range(2)]
    assert ac.inside_jokes(msgs, min_count=4)["jokes"] == []


# --------------------------------------------------------------------------- #
# report rendering                                                             #
# --------------------------------------------------------------------------- #

def test_report_media_table_is_sorted_by_most_media_first(tmp_path):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path)]) == 0
    html = (tmp_path / "saturday_squad" / "report.html").read_text(encoding="utf-8")
    section = re.search(r'<section id="media">.*?</section>', html, re.S)
    assert section, "media section missing from report.html"
    rows = re.findall(r"<tr>(.*?)</tr>", section.group(0), re.S)
    ranked = []
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row, re.S)
        if len(cells) == 7:
            ranked.append((sum(int(c) for c in cells[1:]), cells[0]))
    assert len(ranked) >= 2, f"expected several members, got {ranked}"
    totals = [t for t, _ in ranked]
    assert totals == sorted(totals, reverse=True), f"not descending: {ranked}"
    # Ties fall back to name, so a set's iteration order cannot reshuffle
    # equal rows from one run to the next.
    assert ranked == sorted(ranked, key=lambda r: (-r[0], r[1])), f"unstable order: {ranked}"


def test_media_leaderboard_order_matches_between_markdown_and_html(tmp_path):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path)]) == 0
    out = tmp_path / "saturday_squad"
    md = out / "summary.md"
    html = out / "report.html"
    md_section = md.read_text(encoding="utf-8").split("## Media leaderboard")[1]
    md_section = md_section.split("\n## ")[0]          # stop at the next section
    md_names = [line.split("|")[1].strip()
                for line in md_section.splitlines()
                if line.startswith("|") and not line.startswith("|---")][1:]
    section = re.search(r'<section id="media">.*?</section>',
                        html.read_text(encoding="utf-8"), re.S).group(0)
    html_names = [re.findall(r"<td>(.*?)</td>", row, re.S)[0]
                  for row in re.findall(r"<tr>(.*?)</tr>", section, re.S)
                  if len(re.findall(r"<td>(.*?)</td>", row, re.S)) == 7]
    assert md_names == html_names, f"markdown {md_names} vs html {html_names}"


def test_summary_json_reports_null_for_members_who_never_replied(tmp_path):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path), "--json"]) == 0
    data = json.loads((tmp_path / "saturday_squad" / "summary.json").read_text(encoding="utf-8"))
    assert data["response_speed"], "response speed table is empty"
    for row in data["response_speed"]:
        if row["replies"] == 0:
            assert row["median_seconds"] is None


# --------------------------------------------------------------------------- #
# media paths and incremental state                                            #
# --------------------------------------------------------------------------- #

def test_resolve_media_path_rejects_paths_outside_the_thread(tmp_path):
    thread = tmp_path / "thread"
    (thread / "photos").mkdir(parents=True)
    inside = thread / "photos" / "ok.jpg"
    inside.write_bytes(b"x")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    assert ac.resolve_media_path(thread, "photos/ok.jpg") == inside.resolve()
    assert ac.resolve_media_path(thread, str(outside.resolve())) is None
    assert ac.resolve_media_path(thread, "../secret.txt") is None


def test_resolve_media_path_handles_root_relative_uris(tmp_path):
    """Real exports store uris relative to the export root, not the thread."""
    thread = tmp_path / "your_facebook_activity" / "messages" / "inbox" / "groupchat_123"
    (thread / "photos").mkdir(parents=True)
    (thread / "photos" / "x.jpg").write_bytes(b"x")
    uri = "your_facebook_activity/messages/inbox/groupchat_123/photos/x.jpg"
    assert ac.resolve_media_path(thread, uri) == (thread / "photos" / "x.jpg").resolve()


def test_resolve_media_path_handles_a_renamed_thread_folder(tmp_path):
    thread = tmp_path / "renamed"
    (thread / "gifs").mkdir(parents=True)
    (thread / "gifs" / "g.gif").write_bytes(b"g")
    uri = "your_facebook_activity/messages/inbox/groupchat_123/gifs/g.gif"
    assert ac.resolve_media_path(thread, uri) == (thread / "gifs" / "g.gif").resolve()


def test_check_finds_media_with_root_relative_uris(tmp_path, capsys):
    thread = tmp_path / "your_facebook_activity" / "messages" / "inbox" / "groupchat_123"
    (thread / "photos").mkdir(parents=True)
    (thread / "photos" / "x.jpg").write_bytes(b"x")
    uri = "your_facebook_activity/messages/inbox/groupchat_123/photos/x.jpg"
    (thread / "message_1.json").write_text(json.dumps({
        "title": "groupchat", "messages": [
            {"id": "1", "sender_name": "Alice", "timestamp_ms": 1609459200000,
             "photos": [{"uri": uri}]},
            {"id": "2", "sender_name": "Bob", "timestamp_ms": 1609459200001,
             "photos": [{"uri": uri.replace("x.jpg", "gone.jpg")}]},
        ]}), encoding="utf-8")
    assert ac.main(["--input", str(thread), "--output", str(tmp_path / "out"),
                    "--check"]) == 0
    out = capsys.readouterr().out
    assert "media attachments: 2 | missing on disk: 1" in out, out


def test_top_controls_the_leaderboard_length(tmp_path):
    """--top was documented as controlling leaderboards but was hardcoded to 10,
    so members past the tenth never appeared."""
    def leaderboard(out):
        section = (out / "saturday_squad" / "summary.md").read_text(
            encoding="utf-8").split("## Leaderboard")[1].split("\n## ")[0]
        return [l for l in section.splitlines()
                if l.startswith("|") and not l.startswith("|---")][1:]

    small = tmp_path / "small"
    assert ac.main(["--input", SAMPLE, "--output", str(small), "--top", "2"]) == 0
    assert len(leaderboard(small)) == 2

    big = tmp_path / "big"
    assert ac.main(["--input", SAMPLE, "--output", str(big), "--top", "4"]) == 0
    assert len(leaderboard(big)) == 4


def test_all_time_totals_are_reported(tmp_path):
    out = tmp_path / "out"
    assert ac.main(["--input", SAMPLE, "--output", str(out), "--json"]) == 0
    text = (out / "saturday_squad" / "summary.md").read_text(encoding="utf-8")
    assert "## All-time totals" in text
    for label in ("Messages", "Words", "Emojis", "Reactions", "Questions asked",
                  "Active days", "Conversations"):
        assert f"**{label}**" in text, f"missing total: {label}"
    data = json.loads((out / "saturday_squad" / "summary.json").read_text(encoding="utf-8"))
    assert data["totals"]["Messages"] == "94"
    assert data["total_words"] > 0
    assert data["active_days"] > 0


def test_all_time_totals_survive_skipped_analyses(tmp_path):
    out = tmp_path / "out"
    assert ac.main(["--input", SAMPLE, "--output", str(out),
                    "--skip", "jokes,sentiment,topics,wordcloud"]) == 0
    text = (out / "saturday_squad" / "summary.md").read_text(encoding="utf-8")
    assert "## All-time totals" in text
    assert "**Messages**: 94" in text


def test_skip_omits_analyses_and_still_writes_every_report(tmp_path):
    out = tmp_path / "out"
    assert ac.main(["--input", SAMPLE, "--output", str(out), "--json",
                    "--skip", "jokes,sentiment,wordcloud,topics"]) == 0
    thread = out / "saturday_squad"
    for name in ("summary.md", "report.html", "year_in_review.html", "summary.json"):
        assert (thread / name).exists(), f"missing {name}"
    text = (thread / "summary.md").read_text(encoding="utf-8")
    assert "## Running jokes" not in text
    assert "## Sentiment" not in text
    assert "## What the chat was about" not in text
    assert "## Leaderboard" in text, "unskipped sections should still be there"
    assert not (thread / "wordcloud.png").exists()
    assert not (thread / "inside_jokes.png").exists()
    data = json.loads((thread / "summary.json").read_text(encoding="utf-8"))
    assert data["running_jokes"] is None
    assert data["sentiment"] is None


def test_skip_warns_on_an_unknown_name(tmp_path, capsys):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path),
                    "--skip", "nonsense"]) == 0
    assert "unknown --skip value" in capsys.readouterr().out


def test_incremental_reruns_when_skip_changes(tmp_path, capsys):
    out = tmp_path / "inc_skip"
    base = ["--input", SAMPLE, "--output", str(out), "--incremental"]
    assert ac.main(base) == 0
    capsys.readouterr()
    assert ac.main(base) == 0
    assert "unchanged since last run" in capsys.readouterr().out
    assert ac.main(base + ["--skip", "jokes"]) == 0
    assert "unchanged since last run" not in capsys.readouterr().out


def test_incremental_rerun_when_track_file_changes(tmp_path, capsys):
    out = tmp_path / "inc"
    terms = tmp_path / "terms.txt"
    terms.write_text("bro\n", encoding="utf-8")
    args = ["--input", SAMPLE, "--output", str(out), "--incremental",
            "--track-file", str(terms)]
    assert ac.main(args) == 0
    capsys.readouterr()
    assert ac.main(args) == 0
    assert "unchanged since last run" in capsys.readouterr().out

    terms.write_text("bro\nshawarma\n", encoding="utf-8")
    assert ac.main(args) == 0
    assert "unchanged since last run" not in capsys.readouterr().out
