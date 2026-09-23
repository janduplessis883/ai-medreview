import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from classification_jev import classify_csv


class ClassifyCsvTests(unittest.TestCase):
    def test_preserves_extra_csv_values_in_named_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "reviews.csv"
            input_path.write_text("free_text\nHelpful staff,unexpected value\n", encoding="utf-8")

            classify_csv(
                input_path,
                input_path,
                classifier=lambda _text: ("Reception Staff Friendly and Helpful", 0.91),
            )

            with input_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["free_text"], "Helpful staff")
        self.assertEqual(rows[0]["extra_csv_value_1"], "unexpected value")

    def test_checkpoints_after_every_100_classifications(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "classified.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["free_text"])
                writer.writeheader()
                writer.writerows({"free_text": f"Review {index}"} for index in range(101))

            calls = 0

            def classifier(_text):
                nonlocal calls
                calls += 1
                if calls == 101:
                    raise RuntimeError("simulated API failure")
                return "Reception Staff Friendly and Helpful", 0.91

            with self.assertRaisesRegex(RuntimeError, "simulated API failure"):
                classify_csv(input_path, output_path, classifier=classifier)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[99]["freetext_jev"], "Reception Staff Friendly and Helpful")
        self.assertEqual(rows[100]["freetext_jev"], "")

    def test_writes_nan_for_missing_text_without_calling_classifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "reviews.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["free_text"])
                writer.writeheader()
                writer.writerows([{"free_text": ""}, {"free_text": "NaN"}, {"free_text": "Helpful staff"}])

            calls = []

            def classifier(text):
                calls.append(text)
                return "Reception Staff Friendly and Helpful", 0.91

            classify_csv(input_path, input_path, classifier=classifier)

            with input_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(calls, ["Helpful staff"])
        self.assertEqual(rows[0]["freetext_jev"], "nan")
        self.assertEqual(rows[0]["freetext_jev_score"], "nan")
        self.assertEqual(rows[1]["freetext_jev"], "nan")
        self.assertEqual(rows[1]["freetext_jev_score"], "nan")
        self.assertEqual(rows[2]["freetext_jev"], "Reception Staff Friendly and Helpful")
        self.assertEqual(rows[2]["freetext_jev_score"], "0.91")

    def test_does_not_print_classification_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "reviews.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["free_text"])
                writer.writeheader()
                writer.writerows([{"free_text": ""}, {"free_text": "Helpful staff"}])

            output = io.StringIO()
            with redirect_stdout(output):
                classify_csv(
                    input_path,
                    input_path,
                    classifier=lambda _text: ("Reception Staff Friendly and Helpful", 0.91),
                )

        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
