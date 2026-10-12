"""Hermetic tests for the evaluation package."""

import json

import pytest

from nova.evaluation import accuracy, exact_match, run_evaluation, write_report


def test_exact_match_normalization():
    assert exact_match("Hello  World", "hello world") is True
    assert exact_match("  spaced\tout\n", "SPACED OUT") is True
    assert exact_match("yes", "no") is False
    assert exact_match("", "x") is False


def test_accuracy():
    assert accuracy([("a", "a"), ("b", "c")]) == 0.5
    assert accuracy([]) == 0.0
    assert accuracy([("Hello", "hello")]) == 1.0


def test_run_evaluation_basic():
    predictions = [
        {"id": "q1", "output": "Paris"},
        {"id": "q2", "output": "London"},
    ]
    expected = [
        {"id": "q1", "expected": "paris"},
        {"id": "q2", "expected": "Berlin"},
    ]
    results = run_evaluation(predictions, expected)
    assert results["total"] == 2
    assert results["correct"] == 1
    assert results["accuracy"] == pytest.approx(0.5)
    assert len(results["items"]) == 2
    assert results["by_split"]["standard"]["total"] == 2
    assert results["by_split"]["challenge"]["total"] == 0


def test_run_evaluation_challenge_split():
    predictions = [
        {"id": "q1", "output": "yes"},
        {"id": "q2", "output": "no"},
    ]
    expected = [
        {"id": "q1", "expected": "yes"},
        {"id": "q2", "expected": "no"},
    ]
    results = run_evaluation(predictions, expected, challenge_ids={"q2"})
    assert results["accuracy"] == pytest.approx(1.0)
    assert results["by_split"]["challenge"] == {
        "total": 1, "correct": 1, "accuracy": 1.0,
    }
    assert results["by_split"]["standard"]["total"] == 1
    assert results["by_split"]["standard"]["correct"] == 1


def test_run_evaluation_unmatched_prediction():
    results = run_evaluation(
        [{"id": "q9", "output": "x"}],
        [{"id": "q1", "expected": "x"}],
    )
    assert results["total"] == 0
    assert results["unmatched"] == 1
    assert results["accuracy"] == 0.0


def test_write_report(tmp_path):
    results = run_evaluation(
        [{"id": "q1", "output": "a"}],
        [{"id": "q1", "expected": "a"}],
    )
    path = write_report(results, tmp_path / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["accuracy"] == pytest.approx(1.0)
