import csv
import tempfile
import unittest
from pathlib import Path

from classification_jev_v3 import PRIMARY_TOPIC_CRITERIA, classify_csv


class ClassifyJevV3Tests(unittest.TestCase):
    def test_classifies_each_review_column_and_marks_missing_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "reviews.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["free_text", "do_better"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"free_text": "The receptionist was helpful", "do_better": ""},
                        {"free_text": "", "do_better": "Answer the phone sooner"},
                    ]
                )

            calls = []

            def classifier(text):
                calls.append(text)
                return "Reception Service", 0.91

            classify_csv(input_path, input_path, classifier=classifier)

            with input_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(calls, ["The receptionist was helpful", "Answer the phone sooner"])
        self.assertEqual(rows[0]["jev_freetext_cat"], "Reception Service")
        self.assertEqual(rows[0]["jev_freetext_score"], "0.91")
        self.assertEqual(rows[0]["jev_dobetter_cat"], "nan")
        self.assertEqual(rows[1]["jev_freetext_cat"], "nan")
        self.assertEqual(rows[1]["jev_dobetter_cat"], "Reception Service")

    def test_checkpoints_after_every_100_classified_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "classified.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["free_text", "do_better"])
                writer.writeheader()
                writer.writerows(
                    {"free_text": f"Review {index}", "do_better": f"Improve {index}"}
                    for index in range(51)
                )

            calls = 0

            def classifier(_text):
                nonlocal calls
                calls += 1
                if calls == 101:
                    raise RuntimeError("simulated API failure")
                return "Reception Service", 0.91

            with self.assertRaisesRegex(RuntimeError, "simulated API failure"):
                classify_csv(input_path, output_path, classifier=classifier)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[49]["jev_freetext_cat"], "Reception Service")
        self.assertEqual(rows[49]["jev_dobetter_cat"], "Reception Service")
        self.assertEqual(rows[50]["jev_freetext_cat"], "")

    def test_uses_the_attached_contrastive_criteria(self):
        self.assertEqual(len(PRIMARY_TOPIC_CRITERIA), 26)
        self.assertIn("Telephone Access", PRIMARY_TOPIC_CRITERIA)
        self.assertIn("not_for", PRIMARY_TOPIC_CRITERIA["Telephone Access"])
        self.assertIn("examples", PRIMARY_TOPIC_CRITERIA["Telephone Access"])


if __name__ == "__main__":
    unittest.main()
