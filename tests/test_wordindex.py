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


def test_examples_never_show_the_same_message_twice():
    """A phrase usually has a handful of uses, so a sample drawn over all of
    them would list the first message again under a second heading."""
    msgs = [mk("Alice", BASE, "full send"),
            mk("Bob", BASE + timedelta(minutes=1), "full send",
               reactions=[("Alice", "\U0001f602")])]
    ex = build(msgs).profile("full send")["examples"]
    assert ex["first"]["index"] == 0
    assert ex["most_reacted"]["index"] == 1
    assert ex["random"] == []


def test_a_single_use_is_shown_once():
    msgs = [mk("Alice", BASE, "unicorn")]
    ex = build(msgs).profile("unicorn")["examples"]
    assert ex["first"]["index"] == 0
    assert ex["most_reacted"] is None
    assert ex["random"] == []


def test_random_examples_are_stable_for_the_same_word():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "bruh %d" % i) for i in range(50)]
    index = build(msgs)
    assert (index.profile("bruh")["examples"]["random"]
            == index.profile("bruh")["examples"]["random"])


# --------------------------------------------------------------------------- #
# phrases                                                                      #
# --------------------------------------------------------------------------- #

def test_a_phrase_only_counts_its_words_side_by_side():
    msgs = [mk("Alice", BASE, "full send tonight"),
            mk("Bob", BASE + timedelta(minutes=1), "send the full file")]
    p = build(msgs).profile("full send")
    assert p["uses"] == 1 and p["messages"] == 1
    assert p["word"] == "full send"
    assert p["is_phrase"] is True
    assert p["first"]["sender"] == "Alice"


def test_a_phrase_whose_words_never_touch_has_no_profile():
    msgs = [mk("Alice", BASE, "send me the full list"),
            mk("Bob", BASE + timedelta(minutes=1), "full of send offs")]
    assert build(msgs).profile("full send") is None


def test_a_phrase_with_an_unknown_word_has_no_profile():
    msgs = [mk("Alice", BASE, "full send")]
    assert build(msgs).profile("full sendoff") is None


def test_a_phrase_repeated_in_one_message_counts_every_time():
    msgs = [mk("Alice", BASE, "full send full send"),
            mk("Bob", BASE + timedelta(minutes=1), "full send")]
    p = build(msgs).profile("full send")
    assert p["uses"] == 3 and p["messages"] == 2


def test_punctuation_between_the_words_does_not_break_a_phrase():
    msgs = [mk("Alice", BASE, "Full, SEND!!")]
    assert build(msgs).profile("  full send  ")["uses"] == 1


def test_a_phrase_can_be_built_out_of_stopwords():
    msgs = [mk("Alice", BASE, "in the end it worked"),
            mk("Bob", BASE + timedelta(minutes=1), "the end")]
    assert build(msgs).profile("the end")["uses"] == 2


def test_a_phrase_of_emoji_is_matched_in_order():
    msgs = [mk("Alice", BASE, "\U0001f602\U0001f602\U0001f602"),
            mk("Bob", BASE + timedelta(minutes=1), "\U0001f602 ok \U0001f602")]
    p = build(msgs).profile("\U0001f602\U0001f602")
    assert p["uses"] == 2 and p["messages"] == 1


def test_a_phrase_mixes_words_and_emoji():
    msgs = [mk("Alice", BASE, "lol \U0001f602 yes"),
            mk("Bob", BASE + timedelta(minutes=1), "\U0001f602 lol")]
    assert build(msgs).profile("lol \U0001f602")["uses"] == 1


def test_a_phrase_profile_carries_the_same_blocks_as_a_word():
    msgs = [mk("Alice", BASE, "full send"),
            mk("Bob", BASE + timedelta(days=2), "full send bro"),
            mk("Cara", BASE + timedelta(days=3), "not here")]
    p = build(msgs).profile("full send")
    assert [r["member"] for r in p["per_member"]] == ["Alice", "Bob"]
    assert p["adoption"][1]["days_after"] == 2
    assert p["adoption"][-1]["member"] == "Cara" and p["adoption"][-1]["first"] is None
    assert p["examples"]["first"]["index"] == 0
    assert p["by_year"] == {2020: 2}
    assert p["variants"] == []


def test_a_phrases_own_words_are_not_its_collocations():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "full send bro") for i in range(5)]
    msgs += [mk("Bob", BASE + timedelta(hours=1, minutes=i), "cricket practice")
             for i in range(20)]
    words = [c["word"] for c in build(msgs).profile("full send")["collocations"]]
    assert "bro" in words
    assert "full" not in words and "send" not in words


def test_alone_pct_counts_messages_that_are_only_the_phrase():
    msgs = [mk("Alice", BASE, "full send"),
            mk("Alice", BASE + timedelta(minutes=1), "full send"),
            mk("Bob", BASE + timedelta(minutes=2), "ok full send then")]
    assert build(msgs).profile("full send")["alone_pct"] == 66.7


def test_suggest_completes_the_last_word_and_keeps_the_rest():
    msgs = [mk("Alice", BASE + timedelta(minutes=i), "full send") for i in range(3)]
    msgs += [mk("Bob", BASE + timedelta(hours=1), "sensible")]
    assert build(msgs).suggest("full sen") == ["full send", "full sensible"]


def test_suggest_ignores_a_trailing_space():
    msgs = [mk("Alice", BASE, "full send")]
    assert build(msgs).suggest("full ") == []
