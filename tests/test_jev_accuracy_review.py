import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jev_accuracy_review import build_report, load_or_create_session, record_decision


class JevAccuracyReviewTests(unittest.TestCase):
    def test_session_balances_the_sample_across_jev_categories(self):
        data = pd.DataFrame(
            {
                "free_text": [f"Review {index}" for index in range(30)],
                "freetext_jev": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_or_create_session(
                data, Path(temp_dir) / "session.json", sample_size=8, seed=42
            )

        sampled_categories = data.loc[session["sample_indices"], "freetext_jev"].value_counts()
        self.assertEqual(sorted(sampled_categories.tolist()), [2, 3, 3])

    def test_session_uses_the_same_random_sample_after_restart(self):
        data = pd.DataFrame(
            {
                "free_text": [f"Review {index}" for index in range(150)],
                "freetext_jev": ["Category" for _ in range(150)],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.json"
            first_session = load_or_create_session(data, session_path, sample_size=100, seed=42)
            second_session = load_or_create_session(data, session_path, sample_size=100, seed=999)

        self.assertEqual(first_session["sample_indices"], second_session["sample_indices"])
        self.assertEqual(len(first_session["sample_indices"]), 100)

    def test_record_decision_persists_an_answer_for_resume(self):
        data = pd.DataFrame(
            {"free_text": ["A review"], "freetext_jev": ["Category"]}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.json"
            session = load_or_create_session(data, session_path, sample_size=1, seed=42)
            record_decision(session, session_path, row_index=0, is_correct=True)
            resumed = load_or_create_session(data, session_path, sample_size=1, seed=42)

        self.assertEqual(resumed["decisions"], {"0": True})

    def test_report_calculates_accuracy_and_pending_count(self):
        session = {
            "sample_indices": [10, 11, 12, 13],
            "decisions": {"10": True, "11": False, "12": True},
        }

        report = build_report(session)

        self.assertEqual(report["reviewed"], 3)
        self.assertEqual(report["correct"], 2)
        self.assertEqual(report["incorrect"], 1)
        self.assertEqual(report["pending"], 1)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
