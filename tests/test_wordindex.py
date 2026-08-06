"""Tests for the word index behind the reader's word explorer.

Each test pins a claim the explorer makes on screen: how often a word was said,
who said it first, what it sits next to. The index itself only maps a word to
message indices, so everything below is really a test of the on-demand walk.
"""
from datetime import timedelta

from wordindex import WordIndex
from test_correctness import BASE, mk


def build(msgs):
    return WordIndex(msgs)


# --------------------------------------------------------------------------- #
# core counts                                                                  #
# --------------------------------------------------------------------------- #

def test_core_counts_separate_uses_from_messages():
    msgs = [mk("Alice", BASE, "bruh bruh what"),
            mk("Bob", BASE + timedelta(minutes=1), "bruh"),
            mk("Bob", BASE + timedelta(minutes=2), "nothing here")]
    p = build(msgs).profile("bruh")
    assert p["uses"] == 3
    assert p["messages"] == 2


def test_unknown_word_has_no_profile():
    msgs = [mk("Alice", BASE, "hello there")]
    assert build(msgs).profile("cricket") is None


def test_per_member_rate_surfaces_quiet_members():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "bruh") for i in range(2)]
    msgs += [mk("Bob", BASE + timedelta(hours=1, minutes=i), "quiet")
             for i in range(98)]
    msgs += [mk("Bob", BASE + timedelta(hours=2), "bruh")]
    rows = {r["member"]: r for r in build(msgs).profile("bruh")["per_member"]}
    assert rows["Alice"]["uses"] == 2
    assert rows["Alice"]["per_1k"] == 1000.0
    assert rows["Bob"]["per_1k"] == 10.1


def test_first_and_last_use_carry_the_message():
    msgs = [mk("Alice", BASE, "bruh first"),
            mk("Bob", BASE + timedelta(days=400), "bruh last")]
    p = build(msgs).profile("bruh")
    assert p["first"]["sender"] == "Alice"
    assert p["first"]["content"] == "bruh first"
    assert p["last"]["sender"] == "Bob"
    assert p["peak_year"] in (2020, 2021)


def test_counts_are_bucketed_by_year_month_and_hour():
    msgs = [mk("Alice", BASE, "bruh"),
            mk("Alice", BASE.replace(year=2021), "bruh")]
    p = build(msgs).profile("bruh")
    assert p["by_year"] == {2020: 1, 2021: 1}
    assert p["by_month"]["2020-01"] == 1
    assert p["by_hour"][12] == 2


def test_emoji_are_indexed_like_words():
    msgs = [mk("Alice", BASE, "haha \U0001f602\U0001f602")]
    assert build(msgs).profile("\U0001f602")["uses"] == 2


def test_urls_and_system_events_are_not_indexed():
    msgs = [mk("Alice", BASE, "look https://youtube.com/watch"),
            mk("Bob", BASE + timedelta(minutes=1), "Bob named the group youtube.")]
    assert build(msgs).profile("youtube") is None


def test_a_members_name_is_searchable():
    """Topics drop names because a name is not a topic. Looking one up is the
    whole point of the explorer, so the index must keep them."""
    msgs = [mk("Alice", BASE, "ask alice about it"),
            mk("Bob", BASE + timedelta(minutes=1), "alice said no")]
    assert build(msgs).profile("alice")["uses"] == 2


def test_a_word_said_once_still_has_a_profile():
    msgs = [mk("Alice", BASE, "unicorn"), mk("Bob", BASE + timedelta(minutes=1), "hi")]
    p = build(msgs).profile("unicorn")
    assert p["uses"] == 1 and p["messages"] == 1
    assert p["first"]["content"] == p["last"]["content"]


# --------------------------------------------------------------------------- #
# variants and autocomplete                                                    #
# --------------------------------------------------------------------------- #

def test_variants_are_the_same_word_with_letters_held_down():
    msgs = [mk("Alice", BASE, "bruh"),
            mk("Alice", BASE + timedelta(minutes=1), "bruhh"),
            mk("Bob", BASE + timedelta(minutes=2), "bruhhhh"),
            mk("Bob", BASE + timedelta(minutes=3), "brhu")]
    variants = {v["word"]: v["uses"] for v in build(msgs).variants_of("bruh")}
    assert variants == {"bruhh": 1, "bruhhhh": 1}


def test_folding_variants_adds_them_to_the_totals():
    msgs = [mk("Alice", BASE, "bruh"),
            mk("Alice", BASE + timedelta(minutes=1), "bruhh"),
            mk("Bob", BASE + timedelta(minutes=2), "bruhh")]
    index = build(msgs)
    assert index.profile("bruh")["uses"] == 1
    assert index.profile("bruh", fold_variants=True)["uses"] == 3
    assert index.profile("bruh", fold_variants=True)["messages"] == 3


def test_suggest_ranks_by_how_often_the_word_is_used():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "cricket") for i in range(5)]
    msgs += [mk("Bob", BASE + timedelta(hours=1), "crime")]
    assert build(msgs).suggest("cri") == ["cricket", "crime"]


# --------------------------------------------------------------------------- #
# adoption and reactions                                                       #
# --------------------------------------------------------------------------- #

def test_adoption_order_shows_who_caught_it_from_whom():
    msgs = [mk("Alice", BASE, "bruh"),
            mk("Bob", BASE + timedelta(days=19), "bruh"),
            mk("Cara", BASE + timedelta(days=1), "nothing")]
    adoption = build(msgs).profile("bruh")["adoption"]
    assert [a["member"] for a in adoption] == ["Alice", "Bob", "Cara"]
    assert adoption[0]["days_after"] == 0
    assert adoption[1]["days_after"] == 19
    assert adoption[2]["first"] is None


def test_reaction_pull_compares_against_the_chat_average():
    msgs = [mk("Alice", BASE, "bruh", reactions=[("Bob", "\U0001f602")] * 4),
            mk("Bob", BASE + timedelta(minutes=1), "plain"),
            mk("Bob", BASE + timedelta(minutes=2), "plain"),
            mk("Bob", BASE + timedelta(minutes=3), "plain")]
    # chat mean is 1.0 reactions/message, the word's messages average 4.0
    assert build(msgs).profile("bruh")["reaction_pull"] == 4.0


def test_reaction_pull_is_none_when_the_chat_has_no_reactions():
    msgs = [mk("Alice", BASE, "bruh"), mk("Bob", BASE + timedelta(minutes=1), "hi")]
    assert build(msgs).profile("bruh")["reaction_pull"] is None


def test_adoption_in_a_one_person_chat_is_just_that_person():
    msgs = [mk("Alice", BASE, "bruh"), mk("Alice", BASE + timedelta(days=3), "bruh")]
    adoption = build(msgs).profile("bruh")["adoption"]
    assert len(adoption) == 1
    assert adoption[0]["member"] == "Alice" and adoption[0]["days_after"] == 0


# --------------------------------------------------------------------------- #
# collocations                                                                 #
# --------------------------------------------------------------------------- #

def test_collocations_beat_what_chance_would_predict():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "bruh moment") for i in range(5)]
    msgs += [mk("Bob", BASE + timedelta(hours=1, minutes=i), "cricket practice")
             for i in range(20)]
    colls = {c["word"]: c for c in build(msgs).profile("bruh")["collocations"]}
    assert "moment" in colls
    assert colls["moment"]["ratio"] > 1
    assert "cricket" not in colls


def test_stopwords_are_not_collocations():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "bruh the moment")
            for i in range(5)]
    words = [c["word"] for c in build(msgs).profile("bruh")["collocations"]]
    assert "the" not in words


def test_alone_pct_counts_messages_that_are_only_the_word():
    msgs = [mk("Alice", BASE, "bruh"),
            mk("Alice", BASE + timedelta(minutes=1), "bruh"),
            mk("Bob", BASE + timedelta(minutes=2), "bruh what is this")]
    assert build(msgs).profile("bruh")["alone_pct"] == 66.7


# --------------------------------------------------------------------------- #
# example messages                                                             #
# --------------------------------------------------------------------------- #

def test_examples_include_the_most_reacted_use():
    msgs = [mk("Alice", BASE, "bruh one"),
            mk("Bob", BASE + timedelta(minutes=1), "bruh two",
               reactions=[("Alice", "\U0001f602")] * 3)]
    ex = build(msgs).profile("bruh")["examples"]
    assert ex["first"]["content"] == "bruh one"
    assert ex["most_reacted"]["content"] == "bruh two"


def test_examples_carry_the_message_index_for_jumping():
    msgs = [mk("Alice", BASE, "filler"),
            mk("Alice", BASE + timedelta(minutes=1), "bruh")]
    assert build(msgs).profile("bruh")["examples"]["first"]["index"] == 1


def test_most_reacted_is_none_without_reactions():
    msgs = [mk("Alice", BASE, "bruh")]
    assert build(msgs).profile("bruh")["examples"]["most_reacted"] is None


def test_random_examples_are_stable_for_the_same_word():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "bruh %d" % i) for i in range(50)]
    index = build(msgs)
    assert (index.profile("bruh")["examples"]["random"]
            == index.profile("bruh")["examples"]["random"])
