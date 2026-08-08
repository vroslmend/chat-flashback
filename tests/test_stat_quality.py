"""Tests for the statistics-quality fixes.

Every case here comes from a real number the report got wrong on a 1.8M-message
export: copy-paste floods deciding the vocabulary, Messenger's own group and
nickname events counted as things people typed, skin-tone modifiers ranked as
emoji, and a bot winning the human leaderboards.
"""
from collections import Counter
from datetime import datetime, timedelta

import analyze_chat as ac
from test_correctness import BASE, mk


# --------------------------------------------------------------------------- #
# copy-paste floods                                                            #
# --------------------------------------------------------------------------- #

def test_repeated_word_is_capped_for_vocabulary_but_not_for_volume():
    m = mk("A", BASE, "optix " * 40)
    ac.add_derived_fields([m])
    assert len(m["tokens"]) == 40
    assert m["vocab"].count("optix") == ac.MAX_TOKEN_REPEATS


def test_word_counts_use_the_capped_tokens():
    msgs = [mk("A", BASE, "optix " * 40), mk("B", BASE, "optix")]
    ac.add_derived_fields(msgs)
    stats = ac.core_stats(msgs)
    assert stats["words"]["optix"] == ac.MAX_TOKEN_REPEATS + 1


def test_total_words_still_counts_every_repeat():
    msgs = [mk("A", BASE, "optix " * 40)]
    ac.add_derived_fields(msgs)
    assert ac.core_stats(msgs)["total_words"] == 40


def test_emoji_counts_are_capped_per_message():
    msgs = [mk("A", BASE, "\U0001f600" * 50)]
    ac.add_derived_fields(msgs)
    assert ac.core_stats(msgs)["emojis"]["\U0001f600"] == ac.MAX_TOKEN_REPEATS


def test_unrepeated_message_keeps_the_same_token_tuple():
    """Capping must not double the memory of an ordinary message."""
    m = mk("A", BASE, "a normal sentence")
    ac.add_derived_fields([m])
    assert m["vocab"] is m["tokens"]


def test_flood_messages_are_counted_per_member():
    msgs = [mk("A", BASE, "lol " * ac.FLOOD_REPEATS),
            mk("B", BASE, "lol lol hi")]
    ac.add_derived_fields(msgs)
    stats = ac.core_stats(msgs)
    assert stats["floods"]["A"] == 1
    assert stats["floods"]["B"] == 0


def test_topic_words_are_not_decided_by_one_pasted_message():
    """One paste of 40 `optix` must not outrank a word 10 people actually used."""
    msgs = [mk("A", BASE, "optix " * 40)]
    msgs += [mk("B", BASE + timedelta(minutes=i), "cricket") for i in range(1, 11)]
    ac.add_derived_fields(msgs)
    words = [w["word"] for w in ac.topic_words(msgs)["by_year"][BASE.year]]
    assert words[0] == "cricket"


def test_signature_word_is_not_decided_by_one_pasted_message():
    msgs = [mk("A", BASE, "everyonedicks " * 40)]
    msgs += [mk("A", BASE + timedelta(minutes=i), "kasmein") for i in range(1, 11)]
    msgs += [mk("B", BASE + timedelta(hours=i), "hello there") for i in range(1, 4)]
    ac.add_derived_fields(msgs)
    assert ac.personalities(msgs)["A"]["signature"] == "kasmein"


# --------------------------------------------------------------------------- #
# Messenger's own event messages                                               #
# --------------------------------------------------------------------------- #

def test_group_and_nickname_events_are_not_vocabulary():
    for text in [
        "Ammar Hassan named the group everyonedicks.",
        "Ali Arfa set the nickname for Usman Tahir to British.",
        "Ali Arfa set your nickname to the nalaik.",
        "A contact cleared the nickname for Ali Arfa.",
        "Rafay Ali Awan changed the group photo.",
        "Hashim Cheema changed the theme to Love.",
        "Hashim Cheema removed Rafay Ali Awan from the group.",
        "Rafay Ali Awan left the group.",
        "Ammar Hassan added Jawad Shahid to the group.",
        "You set the quick reaction to \U0001f602.",
        "Ammar Hassan created a poll: Movie.",
        "This poll is no longer available.",
        'Ali Arfa voted for "Option 1" in the poll.',
        "You pinned a message.",
        "A Messenger user started a call.",
        "A contact joined the video call.",
        "The video call ended.",
    ]:
        assert ac._words_only(text) == "", text


def test_ordinary_messages_that_merely_contain_those_verbs_survive():
    for text in [
        "tahts why they added black flash in the final fight in movie",
        "but we turned off da switch ez",
        "he left the group chat vibe entirely lol",
        "i changed the settings on my phone",
        "w light and air pollution",
    ]:
        assert ac._words_only(text) != "", text


def test_event_messages_do_not_become_running_jokes():
    msgs = []
    for i in range(6):
        day = (BASE if i % 2 == 0 else BASE.replace(year=2021)) + timedelta(minutes=i)
        who = "A" if i % 2 else "B"
        msgs.append(mk(who, day, f"{who} named the group thing{i}."))
        msgs.append(mk(who, day + timedelta(seconds=30), "gujjar town"))
    ac.add_derived_fields(msgs)
    phrases = [j["phrase"] for j in ac.inside_jokes(msgs, min_count=3)["jokes"]]
    assert "named group" not in phrases
    assert "gujjar town" in phrases


def test_name_filter_covers_people_who_never_sent_a_message():
    """A member whose account was deleted sends nothing, but is still a name."""
    msgs = [mk("Alice Smith", BASE, "hi")]
    assert ac._member_name_words(msgs, ["Ahsan Raza"]) == {
        "alice", "smith", "ahsan", "raza"}


def test_inside_jokes_record_what_each_year_did():
    msgs = [mk("A", BASE + timedelta(minutes=i), "gujjar town") for i in range(3)]
    msgs += [mk("B", BASE.replace(year=2021) + timedelta(minutes=i), "gujjar town")
             for i in range(5)]
    ac.add_derived_fields(msgs)
    joke = ac.inside_jokes(msgs, min_count=3)["jokes"][0]
    assert joke["count"] == 8
    assert joke["by_year"] == {2020: 3, 2021: 5}
    assert joke["members_by_year"] == {2020: ["A"], 2021: ["B"]}


def test_year_page_shows_that_years_joke_count_not_the_lifetime_one():
    joke = {"phrase": "gujjar town", "count": 8, "members": ["A", "B"],
            "years": [2020, 2021], "by_year": {2020: 3, 2021: 5},
            "members_by_year": {2020: ["A"], 2021: ["B"]}, "example": None}
    page = ac._year_page_html("t", 2020, {}, {"jokes": {"jokes": [joke]}}, [],
                              {"report.html", "year_in_review.html"})
    assert "<td>3</td>" in page
    assert "<td>8</td>" not in page
    assert "<td>A</td>" in page
    assert "<td>A, B</td>" not in page


# --------------------------------------------------------------------------- #
# emoji clusters                                                               #
# --------------------------------------------------------------------------- #

def test_skin_tone_modifier_is_not_counted_as_its_own_emoji():
    assert ac.split_emojis("✋\U0001f3fb") == ["✋\U0001f3fb"]


def test_zwj_sequence_counts_as_one_emoji():
    assert ac.split_emojis("\U0001f926‍♂️") == ["\U0001f926‍♂️"]


def test_plain_emoji_still_split_individually():
    assert ac.split_emojis("hi \U0001f600\U0001f602 there") == ["\U0001f600", "\U0001f602"]


# --------------------------------------------------------------------------- #
# bots                                                                         #
# --------------------------------------------------------------------------- #

def test_meta_ai_is_recognised_as_a_bot():
    assert ac.is_bot("Meta AI")
    assert ac.is_bot("meta ai")
    assert not ac.is_bot("Ammar Hassan")
    assert not ac.is_bot("Meta Ahmed")


def test_fastest_replier_headline_skips_bots():
    speed = {"table": [
        {"member": "Meta AI", "median_s": 1, "median_m": 0.0},
        {"member": "Ammar Hassan", "median_s": 6, "median_m": 0.1},
    ]}
    assert ac._fastest_replier(speed)["member"] == "Ammar Hassan"


def test_weirdest_statements_leave_bots_out():
    msgs = [mk("Meta AI", BASE, "OH YOU WANT A PICTURE?! " * 20),
            mk("Ammar Hassan", BASE, "HALT HALT HALT HALT HALT HALT HALT!!!" * 8)]
    weird = ac.weird_statements(msgs)
    assert [w["member"] for w in weird] == ["Ammar Hassan"]


def test_bot_members_are_labelled_in_tables():
    assert ac.member_label("Meta AI") == "Meta AI (bot)"
    assert ac.member_label("Ammar Hassan") == "Ammar Hassan"


# --------------------------------------------------------------------------- #
# word clouds                                                                  #
# --------------------------------------------------------------------------- #

def test_word_clouds_go_to_the_busiest_members_not_the_first_alphabetically():
    per_member = {"Aaron": Counter({"x": 1}), "Zoe": Counter({"y": 500}),
                  "Mia": Counter({"z": 50})}
    member_msgs = Counter({"Zoe": 500, "Mia": 50, "Aaron": 1})
    assert ac._wordcloud_members(per_member, member_msgs, limit=2) == ["Zoe", "Mia"]


# --------------------------------------------------------------------------- #
# durations                                                                    #
# --------------------------------------------------------------------------- #

def test_short_reply_times_are_reported_in_seconds():
    assert ac._fmt_duration(6) == "6 s"
    assert ac._fmt_duration(0.4) == "0.4 s"


def test_long_reply_times_switch_units():
    assert ac._fmt_duration(90) == "1.5 min"
    assert ac._fmt_duration(7200) == "2.0 h"
    assert ac._fmt_duration(None) == "-"


# --------------------------------------------------------------------------- #
# questions                                                                    #
# --------------------------------------------------------------------------- #

def test_one_reply_to_several_queued_questions_counts_as_one_response():
    msgs = [mk("A", BASE, "where?"),
            mk("A", BASE + timedelta(seconds=10), "when?"),
            mk("A", BASE + timedelta(seconds=20), "why?"),
            mk("B", BASE + timedelta(seconds=30), "no idea")]
    q = ac.question_stats(msgs)
    row = next(r for r in q["table"] if r["member"] == "B")
    assert row["responses_given"] == 1
    assert q["total_answered"] == 3
    # One reply is one data point, timed from the question that waited longest.
    assert row["median_s"] == 30


# --------------------------------------------------------------------------- #
# percentages                                                                  #
# --------------------------------------------------------------------------- #

def test_the_same_percentage_is_reported_everywhere():
    assert ac._pct(41466, 42003) == 99
    assert ac._pct(0, 0) == 0
