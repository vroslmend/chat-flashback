from pathlib import Path

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")


def run_pipeline(tmp_path, *extra):
    out = tmp_path / "out"
    return out, ac.main(["--input", SAMPLE, "--output", str(out), *extra])


def test_full_pipeline_produces_report_and_charts(tmp_path):
    out, code = run_pipeline(tmp_path, "--track", "shawarma, bro")
    assert code == 0
    report = out / "saturday_squad" / "summary.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Total messages" in text
    assert "94" in text
    assert "Yearly recaps" in text
    assert "Member personalities" in text
    assert "Reaction dynamics" in text
    assert "Response speed" in text
    assert "Swear-word analytics" in text
    assert "Weirdest statements" in text
    assert "shawarma" in text
    assert "bro" in text
    assert "shit" in text or "hell" in text or "damn" in text
    assert "Period" in text and "2017" in text and "2026" in text
    for chart in ["messages_by_year.png", "activity_by_hour.png", "activity_by_weekday.png",
                  "top_members.png", "top_words.png", "yearly_recap.png",
                  "reactions_given.png", "most_reacted.png", "response_speed.png",
                  "swear_by_member.png", "swear_over_time.png", "tracked_terms.png",
                  "top_emojis.png"]:
        assert (out / "saturday_squad" / chart).exists(), f"missing {chart}"


def test_anonymize_removes_names(tmp_path):
    out, code = run_pipeline(tmp_path, "--anonymize")
    assert code == 0
    text = (out / "saturday_squad" / "summary.md").read_text(encoding="utf-8")
    assert "Person A" in text
    for name in ["Alice", "Bob", "Charlie", "Dana"]:
        assert name not in text


def test_year_filter_limits_period(tmp_path):
    out, code = run_pipeline(tmp_path, "--year", "2021")
    assert code == 0
    text = (out / "saturday_squad" / "summary.md").read_text(encoding="utf-8")
    assert "2021" in text
    assert "2017" not in text


def test_no_track_no_tracked_terms_section(tmp_path):
    out, code = run_pipeline(tmp_path)
    assert code == 0
    text = (out / "saturday_squad" / "summary.md").read_text(encoding="utf-8")
    assert "Custom tracked terms" not in text
    assert not (out / "saturday_squad" / "tracked_terms.png").exists()
