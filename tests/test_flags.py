import json
from pathlib import Path

import pytest

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")


def test_tz_flag_shifts_hours(tmp_path):
    out = tmp_path / "a"
    assert ac.main(["--input", SAMPLE, "--output", str(out), "--tz", "+03:00", "--json"]) == 0
    out2 = tmp_path / "b"
    assert ac.main(["--input", SAMPLE, "--output", str(out2), "--tz", "-05:00", "--json"]) == 0
    json_a = json.loads((out / "saturday_squad" / "summary.json").read_text(encoding="utf-8"))
    json_b = json.loads((out2 / "saturday_squad" / "summary.json").read_text(encoding="utf-8"))
    assert json_a["total_messages"] == json_b["total_messages"] == 94


def test_bad_tz_flag_skips_cleanly(tmp_path, capsys):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path), "--tz", "bogus"]) == 0
    err = capsys.readouterr().out
    assert "invalid timezone" in err


def test_config_file_merge(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"output": str(tmp_path / "cfg_out"), "top": 3, "json": True}),
                   encoding="utf-8")
    assert ac.main(["--input", SAMPLE, "--config", str(cfg)]) == 0
    text = (tmp_path / "cfg_out" / "saturday_squad" / "summary.md").read_text(encoding="utf-8")
    assert "| Alice | 26" in text
    assert (tmp_path / "cfg_out" / "saturday_squad" / "summary.json").exists()


def test_cli_overrides_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"top": 3, "output": str(tmp_path / "cfg_out2")}),
                   encoding="utf-8")
    assert ac.main(["--input", SAMPLE, "--config", str(cfg), "--top", "5"]) == 0


def test_bad_config_is_ignored(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text("not json", encoding="utf-8")
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path), "--config", str(cfg)]) == 0
    assert "cannot read config" in capsys.readouterr().out


def test_progress_flag_smoke(tmp_path):
    assert ac.main(["--input", SAMPLE, "--output", str(tmp_path), "--progress"]) == 0
    assert (tmp_path / "saturday_squad" / "summary.md").exists()


def test_incremental_skips_unchanged(tmp_path, capsys):
    out = tmp_path / "inc"
    assert ac.main(["--input", SAMPLE, "--output", str(out), "--incremental", "--json"]) == 0
    state = json.loads((out / ".chatflashback_state.json").read_text(encoding="utf-8"))
    assert state
    assert ac.main(["--input", SAMPLE, "--output", str(out), "--incremental", "--json"]) == 0
    out_log = capsys.readouterr().out
    assert "unchanged since last run" in out_log


def test_incremental_reprocesses_when_file_changes(tmp_path, capsys):
    out = tmp_path / "inc2"
    assert ac.main(["--input", SAMPLE, "--output", str(out), "--incremental"]) == 0
    capsys.readouterr()
    target = next(Path(SAMPLE).glob("message_*.json"))
    orig = target.read_bytes()
    try:
        target.write_bytes(orig + b" ")
        assert ac.main(["--input", SAMPLE, "--output", str(out), "--incremental"]) == 0
        assert "unchanged since last run" not in capsys.readouterr().out
    finally:
        target.write_bytes(orig)
