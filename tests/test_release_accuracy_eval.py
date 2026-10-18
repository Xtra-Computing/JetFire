import unittest

from jetfire_vllm.bench.longbench_metrics import classification_score
from jetfire_vllm.bench.release_accuracy_eval import (
    _score_longbench_v1_prediction,
    _score_niah,
)


class NiahScoringTest(unittest.TestCase):
    def test_exact_number(self) -> None:
        self.assertTrue(_score_niah("3345947", "3345947"))

    def test_number_with_internal_separators(self) -> None:
        self.assertTrue(_score_niah("3,345,947", "3345947"))
        self.assertTrue(_score_niah("334,5947", "3345947"))
        self.assertTrue(_score_niah("3_345_947", "3345947"))

    def test_whitespace_is_removed_from_entire_output(self) -> None:
        self.assertTrue(_score_niah("334 5947", "3345947"))

    def test_rejects_explanation_and_extra_numbers(self) -> None:
        self.assertFalse(_score_niah("The number is 3345947", "3345947"))
        self.assertFalse(_score_niah("3345947 or 123", "3345947"))

    def test_rejects_target_embedded_in_larger_number(self) -> None:
        self.assertFalse(_score_niah("133459470", "3345947"))

    def test_wrong_number(self) -> None:
        self.assertFalse(_score_niah("334,947", "3345947"))


class LongBenchScoringTest(unittest.TestCase):
    def test_trec_uses_only_first_output_line(self) -> None:
        sample = {
            "answers": ["Individual"],
            "all_classes": ["Individual", "Date"],
        }
        score = _score_longbench_v1_prediction(
            "trec",
            "Individual\nDate",
            sample,
            classification_score,
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
