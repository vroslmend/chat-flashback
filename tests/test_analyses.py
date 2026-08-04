from pathlib import Path

import pytest

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")


def load_sample():
    return ac.normalize_messages(load_raw())


def load_raw():
    import json as _json
    msgs = []
    for i in range(1, 5):
        data = _json.load(open(f"{SAMPLE}/message_{i}.json", encoding="utf-8"))
        msgs.extend(data["messages"])
    return msgs


def test_activity_heatmap_shape():
    hm = ac.activity_heatmap(load_sample())
    assert hm is not None
    assert hm["active_days"] >= 20
    assert all(len(w) == 7 for w in hm["weeks"])
    assert hm["max_day"] >= 7


def test_pace_trends_rolling():
    pace = ac.pace_trends(load_sample())
    assert pace["days"]
    assert len(pace["rolling"]) == len(pace["days"])
    assert all(r >= 0 for r in pace["rolling"])
    assert "media_by_year" in pace


def test_pair_matrices():
    pm = ac.pair_matrices(load_sample())
    assert "Alice" in pm["members"]
    assert pm["reply"]["Bob"].get("Alice", 0) >= 10
    assert pm["reaction"]["Charlie"].get("Bob", 0) >= 0


def test_hourly_radar_24_bins():
    radar = ac.hourly_radar(load_sample())
    assert radar
    assert all(len(h) == 24 for h in radar.values())


def test_word_cloud_data():
    wc = ac.word_cloud_data(load_sample())
    assert wc["overall"]
    assert wc["per_member"]["Alice"]


def test_monologues():
    mono = ac.monologues(load_sample())
    assert mono["longest_run_len"] >= 3
    assert mono["email_moments"] == 0 or mono["per_member_longest"]
    assert mono["longest_run"][0]["sender"] in mono["per_member_longest"]


def test_unsent_stats_empty():
    unsent = ac.unsent_stats(load_sample())
    assert sum(unsent.values()) == 0


def test_insights_built():
    msgs = load_sample()
    stats = ac.core_stats(msgs)
    analyses = {
        "conversations": ac.conversation_starters(msgs),
        "speed": ac.response_speed(msgs),
        "ghosting": ac.ghosting(msgs),
        "monologues": ac.monologues(msgs),
        "pair_matrices": ac.pair_matrices(msgs),
        "sentiment": ac.sentiment_analysis(msgs),
        "extremes": ac.extremes(msgs),
    }
    out = ac.insights(stats, analyses)
    assert isinstance(out, list)
    assert out


def test_wordcloud_chart_generated(tmp_path):
    pytest.importorskip("wordcloud")
    code = ac.main(["--input", SAMPLE, "--output", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "saturday_squad" / "wordcloud.png").exists()


def test_new_charts_exist(tmp_path):
    code = ac.main(["--input", SAMPLE, "--output", str(tmp_path)])
    assert code == 0
    for chart in ["activity_heatmap.png", "pace_trends.png", "reply_matrix.png",
                  "reaction_matrix.png", "hourly_radar.png", "monologues.png"]:
        assert (tmp_path / "saturday_squad" / chart).exists(), f"missing {chart}"


def test_emoji_stats():
    emo = ac.emoji_stats(load_sample())
    assert emo["total_emojis"] > 0
    assert emo["per_member"]
    assert emo["per_year"]
    assert all(p >= 0 for p in emo["emojis_per_100"].values())
    assert sum(sum(c.values()) for c in emo["per_year"].values()) == emo["total_emojis"]


def test_question_stats():
    q = ac.question_stats(load_sample())
    assert q["total_questions"] >= 1
    assert q["table"]
    assert sum(r["asked"] for r in q["table"]) == q["total_questions"]
    assert 0 <= q["total_answered"] <= q["total_questions"]


def test_topic_words():
    topics = ac.topic_words(load_sample())
    assert topics["by_year"]
    assert topics["years"] == sorted(topics["years"])
    words = topics["by_year"][topics["years"][0]]
    assert words and all(w["count"] > 0 for w in words)
    assert all(len(w["word"]) > 2 for w in words)


def test_inside_jokes_synthetic():
    from datetime import datetime, timedelta
    base = datetime(2020, 1, 1)
    msgs = []
    n = 0
    for year in range(2020, 2023):
        for sender in ("Alice", "Bob", "Charlie"):
            for _ in range(3):
                ts = int((base.replace(year=year) + timedelta(days=n)).timestamp() * 1000)
                msgs.append({"sender": sender, "ts_ms": ts, "content": "pizza friday night party",
                             "mtype": "Generic", "reactions": [], "has_photo": False,
                             "has_sticker": False, "photo_uris": [], "link": None,
                             "call_duration": None, "reply_to": None, "is_unsent": False,
                             "dt": base.replace(year=year) + timedelta(days=n)})
                n += 1
    jokes = ac.inside_jokes(msgs)
    assert jokes["jokes"], "expected at least one running joke"
    top = jokes["jokes"][0]
    assert top["count"] >= 4
    assert len(top["years"]) >= 2
    assert len(top["members"]) >= 2


def test_timezone_shift():
    msgs = load_sample()
    default = ac.core_stats(msgs)["by_hour"]
    shifted = ac.core_stats(ac.normalize_messages(load_raw(), tz="+03:00"))["by_hour"]
    assert sum(default.values()) == sum(shifted.values())


def test_bad_timezone_rejected():
    import pytest
    with pytest.raises(ValueError):
        ac.normalize_messages([], tz="not/a/zone")
