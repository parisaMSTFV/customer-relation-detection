import json
import sys

import pandas as pd
import pytest

from customer_relation_detection import cli


def test_smoke_command_uses_a_temporary_output(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_pipeline", lambda path: calls.append(path) or {"ok": True})
    monkeypatch.setattr(sys, "argv", ["relation-detection", "smoke"])
    cli.main()
    assert calls[0].name.startswith("relation-detection-")
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_reproduce_command_accepts_an_explicit_output(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: "CONFIG")
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda output, config: {"output": str(output), "config": config},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["relation-detection", "reproduce", "--output-root", str(tmp_path)],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["config"] == "CONFIG"


def test_analyze_command_requires_and_uses_salt(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("RELATION_ID_SALT", "1234567890abcdef")
    monkeypatch.setattr(cli.pd, "read_csv", lambda path: pd.DataFrame({"loaded": [str(path)]}))
    monkeypatch.setattr(cli, "load_config", lambda path: "CONFIG")
    monkeypatch.setattr(
        cli,
        "run_analysis",
        lambda orders, output, **kwargs: {
            "rows": len(orders),
            "salt": kwargs["identifier_salt"],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relation-detection",
            "analyze",
            "--input",
            "orders.csv",
            "--output-root",
            str(tmp_path),
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"rows": 1, "salt": "1234567890abcdef"}


def test_analyze_command_fails_closed_without_salt(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("RELATION_ID_SALT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relation-detection",
            "analyze",
            "--input",
            "orders.csv",
            "--output-root",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit, match="Set RELATION_ID_SALT"):
        cli.main()


def test_review_feedback_command_reads_both_inputs(monkeypatch, tmp_path, capsys) -> None:
    loaded = []

    def fake_read_csv(path):
        loaded.append(str(path))
        return pd.DataFrame({"source": [str(path)]})

    monkeypatch.setattr(cli.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        cli,
        "run_review_feedback_audit",
        lambda guardrail, feedback, output: {
            "guardrail_rows": len(guardrail),
            "feedback_rows": len(feedback),
            "output": str(output),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relation-detection",
            "audit-review-feedback",
            "--guardrail",
            "guardrail.csv",
            "--feedback",
            "feedback.csv",
            "--output-root",
            str(tmp_path),
        ],
    )
    cli.main()
    assert loaded == ["guardrail.csv", "feedback.csv"]
    result = json.loads(capsys.readouterr().out)
    assert result["guardrail_rows"] == 1
    assert result["feedback_rows"] == 1
