"""Numeric tests for the chat-level narratives.

Same shape as test_correctness.py: small hand-made message lists whose answers
can be worked out by hand, one test per claim the pages make.
"""
from datetime import timedelta

import chatstats as cs
from test_correctness import BASE, mk


def sysmsg(sender, dt, content):
    return mk(sender, dt, content)


# --------------------------------------------------------------------------- #
# the group's own history                                                      #
# --------------------------------------------------------------------------- #

def test_group_names_form_a_timeline_with_end_dates():
    msgs = [sysmsg("Alice", BASE, "Alice named the group squad goals."),
            mk("Bob", BASE + timedelta(days=1), "hi"),
            sysmsg("Bob", BASE + timedelta(days=30), "Bob named the group the boys.")]
    history = cs.group_history(msgs)
    assert [n["name"] for n in history["names"]] == ["squad goals", "the boys"]
    assert history["names"][0]["until"] == history["names"][1]["date"]
    assert history["names"][-1]["until"] is None
    assert history["current_name"] == "the boys"


def test_a_nickname_is_stored_as_a_range():
    msgs = [sysmsg("Alice", BASE, "Alice set the nickname for Bob to speed racer."),
            sysmsg("Alice", BASE + timedelta(days=60),
                   "Alice set the nickname for Bob to sigma male.")]
    entries = cs.group_history(msgs)["nicknames"]["Bob"]
    assert [e["nickname"] for e in entries] == ["speed racer", "sigma male"]
    assert entries[0]["until"] == entries[1]["from"]
    assert entries[1]["until"] is None


def test_clearing_a_nickname_closes_its_range():
    msgs = [sysmsg("Alice", BASE, "Alice set the nickname for Bob to speed racer."),
            sysmsg("Bob", BASE + timedelta(days=5), "Bob cleared the nickname for Bob.")]
    entry = cs.group_history(msgs)["nicknames"]["Bob"][0]
    assert entry["until"] == (BASE + timedelta(days=5)).strftime("%Y-%m-%d")


def test_membership_events_are_kept_apart_from_renames():
    msgs = [sysmsg("Alice", BASE, "Alice added Cara to the group."),
            sysmsg("Cara", BASE + timedelta(days=1), "Cara left the group."),
            sysmsg("Alice", BASE + timedelta(days=2), "Alice named the group us.")]
    history = cs.group_history(msgs)
    assert [e["kind"] for e in history["membership"]] == ["added", "left"]
    assert history["membership"][0]["target"] == "Cara"
    assert len(history["names"]) == 1


def test_the_busiest_renamer_is_counted_across_every_event_kind():
    msgs = [sysmsg("Alice", BASE, "Alice named the group one."),
            sysmsg("Alice", BASE + timedelta(days=1), "Alice changed the group photo."),
            sysmsg("Bob", BASE + timedelta(days=2), "Bob changed the theme to Love.")]
    history = cs.group_history(msgs)
    assert history["busiest"].most_common(1) == [("Alice", 2)]
    assert history["total"] == 3


def test_ordinary_text_that_reads_like_an_event_is_not_one():
    """SYSTEM_EVENT_RE anchors to the whole message for exactly this reason."""
    assert cs.parse_event("they added black flash in the final fight") is None
    assert cs.parse_event("") is None


def test_a_nickname_set_on_the_exporting_account_is_counted_not_placed():
    msgs = [sysmsg("Alice", BASE, "Alice set your nickname to captain.")]
    history = cs.group_history(msgs)
    assert history["nicknames"] == {}
    assert history["busiest"]["Alice"] == 1


# --------------------------------------------------------------------------- #
# relationships                                                                #
# --------------------------------------------------------------------------- #

def test_a_pair_is_counted_from_replies_and_reactions_together():
    msgs = [mk("Alice", BASE, "hey"),
            mk("Bob", BASE + timedelta(minutes=1), "hi", reactions=[("Alice", "❤")]),
            mk("Alice", BASE + timedelta(minutes=2), "ok")]
    rel = cs.relationships(msgs)
    pair = rel["pairs"][0]
    assert pair["pair"] == ["Alice", "Bob"]
    # two replies inside the window, plus one reaction
    assert pair["total"] == 3


def test_a_reply_after_the_window_is_not_an_interaction():
    msgs = [mk("Alice", BASE, "hey"),
            mk("Bob", BASE + timedelta(hours=5), "hi")]
    assert cs.relationships(msgs)["pairs"] == []


def test_who_speaks_first_after_a_day_of_silence():
    msgs = [mk("Alice", BASE, "hey"),
            mk("Bob", BASE + timedelta(days=3), "anyone alive"),
            mk("Alice", BASE + timedelta(days=3, minutes=1), "yes")]
    assert cs.relationships(msgs)["first_after_silence"] == {"Bob": 1}


def test_the_last_word_of_every_session_is_counted():
    msgs = [mk("Alice", BASE, "hey"),
            mk("Bob", BASE + timedelta(minutes=1), "bye"),
            mk("Alice", BASE + timedelta(hours=4), "back"),
            mk("Bob", BASE + timedelta(hours=4, minutes=1), "hi again")]
    assert cs.relationships(msgs)["last_word"] == {"Bob": 2}


def test_going_unanswered_is_reported_as_a_rate_not_a_count():
    """The loudest member is left unanswered most often just by talking most."""
    msgs = []
    for i in range(16):
        msgs.append(mk("Alice", BASE + timedelta(days=i), "chatter"))
        msgs.append(mk("Bob", BASE + timedelta(days=i, minutes=5), "answer"))
    for i in range(16, 20):
        msgs.append(mk("Alice", BASE + timedelta(days=i), "anyone there"))
    msgs.append(mk("Cara", BASE + timedelta(days=30), "hello?"))
    msgs.append(mk("Cara", BASE + timedelta(days=31), "hello??"))
    rows = {r["member"]: r for r in cs.relationships(msgs)["ignored"]}
    assert rows["Alice"]["unanswered"] == 4 and rows["Alice"]["pct"] == 20.0
    assert rows["Cara"]["unanswered"] == 2 and rows["Cara"]["pct"] == 100.0
    # Alice is ignored more often, Cara is ignored more reliably; the ranking
    # is by rate, so Cara comes first.
    assert rows["Cara"]["unanswered"] < rows["Alice"]["unanswered"]
    assert cs.relationships(msgs)["ignored"][0]["member"] in ("Bob", "Cara")


def test_a_member_who_barely_spoke_is_not_drifting():
    """Two messages in a year make every partner look like half their world."""
    start = BASE.replace(year=2020, month=1, day=1)
    msgs = []
    for i in range(60):
        msgs.append(mk("Alice", start + timedelta(days=i), "hey"))
        msgs.append(mk("Bob", start + timedelta(days=i, minutes=1), "hi"))
        msgs.append(mk("Alice", start + timedelta(days=i, minutes=2), "ok"))
    year2 = start.replace(year=2021)
    for i in range(60):
        msgs.append(mk("Alice", year2 + timedelta(days=i), "hey"))
        msgs.append(mk("Cara", year2 + timedelta(days=i, minutes=1), "hi"))
        msgs.append(mk("Alice", year2 + timedelta(days=i, minutes=2), "ok"))
    # Bob turns up once at the very end and says one thing to Alice.
    msgs.append(mk("Alice", year2.replace(month=12, day=31), "bob?"))
    msgs.append(mk("Bob", year2.replace(month=12, day=31) + timedelta(minutes=1), "here"))
    rel = cs.relationships(msgs)
    assert not [d for d in rel["drift"] if d["member"] == "Bob"]
    # Alice's own side of the same pair still has the numbers to say it.
    assert [d for d in rel["drift"] if d["member"] == "Alice"]


def test_drift_needs_a_full_year_on_both_ends():
    """A pair that only ever talked in a partial year has nothing to compare."""
    msgs = [mk("Alice", BASE, "hey"), mk("Bob", BASE + timedelta(minutes=1), "hi")]
    rel = cs.relationships(msgs)
    assert rel["full_years"] == []
    assert rel["drift"] == []


def test_drift_is_reported_when_a_pairs_share_halves():
    start = BASE.replace(year=2020, month=1, day=1)
    msgs = []
    # 2020: Alice and Bob talk to each other and nobody else. Enough exchanges
    # to clear the floor that keeps three-message pairs off the page.
    for i in range(60):
        msgs.append(mk("Alice", start + timedelta(days=i, minutes=0), "hey"))
        msgs.append(mk("Bob", start + timedelta(days=i, minutes=1), "hi"))
        msgs.append(mk("Alice", start + timedelta(days=i, minutes=2), "ok"))
    # 2021: Alice mostly talks to Cara instead.
    year2 = start.replace(year=2021)
    for i in range(60):
        msgs.append(mk("Alice", year2 + timedelta(days=i, minutes=0), "hey"))
        msgs.append(mk("Cara", year2 + timedelta(days=i, minutes=1), "hi"))
        msgs.append(mk("Alice", year2 + timedelta(days=i, minutes=2), "ok"))
    msgs.append(mk("Bob", year2.replace(month=12, day=31), "still here"))
    msgs.append(mk("Alice", year2.replace(month=12, day=31) + timedelta(minutes=1), "hi"))
    rel = cs.relationships(msgs)
    assert 2021 in rel["full_years"]
    dropped = [d for d in rel["drift"]
               if d["pair"] == ["Alice", "Bob"] and d["member"] == "Alice"]
    assert dropped and dropped[0]["change_pct"] < -50


# --------------------------------------------------------------------------- #
# member arcs                                                                  #
# --------------------------------------------------------------------------- #

def test_a_member_profile_covers_their_own_years_only():
    msgs = [mk("Alice", BASE, "cricket was great"),
            mk("Alice", BASE.replace(year=2021), "exams are brutal"),
            mk("Bob", BASE + timedelta(minutes=1), "cricket cricket cricket")]
    p = cs.member_profile(msgs, "Alice")
    assert p["total"] == 2
    assert p["by_year"] == {2020: 1, 2021: 1}
    assert set(p["words_by_year"]) == {2020, 2021}
    words_2021 = [w["word"] for w in p["words_by_year"][2021]]
    assert "exams" in words_2021 and "cricket" not in words_2021


def test_a_member_profile_knows_who_they_answer():
    msgs = [mk("Alice", BASE, "hey"),
            mk("Bob", BASE + timedelta(minutes=1), "hi"),
            mk("Alice", BASE + timedelta(minutes=2), "ok"),
            mk("Cara", BASE + timedelta(minutes=3), "hello")]
    p = cs.member_profile(msgs, "Alice")
    assert p["closest"][0] == ("Bob", 1)
    assert p["talks_to"][2020] == {"Bob": 1}


def test_a_member_profile_keeps_their_most_reacted_messages():
    msgs = [mk("Alice", BASE, "plain"),
            mk("Alice", BASE + timedelta(minutes=1), "the good one",
               reactions=[("Bob", "\U0001f602")] * 3)]
    p = cs.member_profile(msgs, "Alice")
    assert p["top_reacted"][0]["content"] == "the good one"
    assert len(p["top_reacted"]) == 1


def test_a_member_who_never_spoke_has_no_profile():
    msgs = [mk("Alice", BASE, "hi")]
    assert cs.member_profile(msgs, "Nobody") is None


def test_member_profiles_are_ordered_by_how_much_they_said():
    msgs = [mk("Bob", BASE + timedelta(minutes=i), "hi") for i in range(3)]
    msgs += [mk("Alice", BASE + timedelta(hours=1), "hey")]
    assert list(cs.member_profiles(msgs)) == ["Bob", "Alice"]


# --------------------------------------------------------------------------- #
# eras and vocabulary turnover                                                 #
# --------------------------------------------------------------------------- #

# A quarter needs a full set of top words before two of them can be compared,
# so an era fixture needs a real vocabulary rather than three words. The first
# word is repeated because equally exclusive words are ranked by count, which
# is what makes the era's name predictable.
ERA_A = " ".join(["cricket"] * 3 + """practice tournament season bat ball wicket
pitch stumps runs catch field umpire spin pace toss innings boundary sixes
fours""".split())
ERA_B = " ".join(["wedding"] * 3 + """planning venue booking dress cake guest
invite band flowers rings vows honeymoon suite caterer tables seating speech
photos dances""".split())
PER_MONTH = 15


def _monthly(sender, months, per_month, text):
    """`per_month` messages in each of `months` consecutive months from BASE."""
    out = []
    for n in range(months):
        year = BASE.year + (BASE.month - 1 + n) // 12
        month = (BASE.month - 1 + n) % 12 + 1
        for i in range(per_month):
            out.append(mk(sender, BASE.replace(year=year, month=month, day=1)
                          + timedelta(days=i % 27, minutes=i), text))
    return out


def _two_eras(first_sender="Alice", second_sender="Alice", months=12):
    """`months` of one vocabulary followed by `months` of a different one."""
    first = _monthly(first_sender, months, PER_MONTH, ERA_A)
    second = _monthly(second_sender, months * 2, PER_MONTH, ERA_B)[months * PER_MONTH:]
    return first + second


def test_a_short_chat_is_not_segmented():
    msgs = _monthly("Alice", 4, 3, "hello there")
    result = cs.eras(msgs)
    assert result["eras"] == []
    assert result["reason"]


def test_a_steady_chat_stays_one_era():
    msgs = _monthly("Alice", 24, PER_MONTH, ERA_A)
    result = cs.eras(msgs)
    assert len(result["eras"]) == 1
    assert result["eras"][0]["months"] == 24


def test_a_sparse_chat_is_all_one_era():
    """Three messages against eight is not a collapse, it is a quiet chat."""
    msgs = _monthly("Alice", 36, 2, "hello there everyone")
    assert len(cs.eras(msgs)["eras"]) == 1


def test_a_volume_collapse_opens_a_new_era():
    msgs = _monthly("Alice", 12, 40, ERA_A)
    tail = _monthly("Alice", 24, 4, ERA_A)[12 * 4:]
    result = cs.eras(msgs + tail)
    assert len(result["eras"]) >= 2
    assert result["eras"][0]["start"] == "2020-01"


def test_a_topic_turnover_opens_a_new_era_and_names_it():
    result = cs.eras(_two_eras())
    assert len(result["eras"]) == 2
    assert result["eras"][0]["name"] == "cricket"
    assert result["eras"][1]["name"] == "wedding"


def test_an_era_is_named_for_what_is_its_own_not_what_is_everywhere():
    """The filler word of a chat is in every era, so it names none of them."""
    first = _monthly("Alice", 12, PER_MONTH, "hai " + ERA_A)
    second = _monthly("Alice", 24, PER_MONTH, "hai " + ERA_B)[12 * PER_MONTH:]
    result = cs.eras(first + second)
    assert [e["name"] for e in result["eras"]] == ["cricket", "wedding"]
    assert "hai" not in [w["word"] for w in result["eras"][0]["words"]]


def test_an_era_reports_its_own_volume_and_top_member():
    result = cs.eras(_two_eras(first_sender="Alice", second_sender="Bob"))
    assert result["eras"][0]["top_member"] == "Alice"
    # The rule reads three months ahead, so it calls the change once most of
    # that window has turned over -- a month before the vocabulary fully has.
    assert abs(result["eras"][0]["messages"] - 12 * PER_MONTH) <= PER_MONTH


def test_words_are_born_in_the_year_they_first_appear():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "cricket season") for i in range(6)]
    msgs += [mk("Alice", BASE.replace(year=2021) + timedelta(minutes=i), "wedding season")
             for i in range(6)]
    turnover = cs.vocabulary_turnover(msgs)
    assert [w["word"] for w in turnover["born"][2021]] == ["wedding"]
    assert 2020 not in turnover["born"]


def test_a_word_dies_in_its_last_year_but_never_in_the_final_one():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "cricket season") for i in range(6)]
    msgs += [mk("Alice", BASE.replace(year=2021) + timedelta(minutes=i), "wedding season")
             for i in range(6)]
    turnover = cs.vocabulary_turnover(msgs)
    assert [w["word"] for w in turnover["died"][2020]] == ["cricket"]
    assert 2021 not in turnover["died"]


def test_a_word_said_once_is_not_vocabulary():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "cricket season") for i in range(6)]
    msgs += [mk("Alice", BASE.replace(year=2021), "unicorn season")]
    turnover = cs.vocabulary_turnover(msgs)
    assert turnover["born"].get(2021, []) == []
