import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "wt_a_policy_truth.py"
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "relaunch_handoffs"
    / "WT-A_policy_truth_artifact_2026-04-17.json"
)


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("wt_a_policy_truth_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WTAPolicyTruthArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool_module()
        cls.data = cls.tool.load_artifact(ARTIFACT_PATH)

    def test_validate_artifact_has_no_errors(self):
        self.assertEqual(self.tool.validate_artifact(self.data), [])

    def test_current_training_is_old_semantics(self):
        policy = self.data["policy_awareness"]
        self.assertFalse(policy["current_training_policy_aware"])
        self.assertEqual(
            policy["same_day_labels"]["current_checked_in_track_a_configs"],
            "old-semantics",
        )
        self.assertEqual(
            policy["same_day_labels"]["explicit_hint_lane_retrains_on_current_data"],
            "weak-signal-only",
        )
        self.assertFalse(policy["same_day_labels"]["policy_corrected_available_today"])

    def test_corrected_mixed_p050_counts_match_expected_totals(self):
        family = next(
            family
            for family in self.data["classification_families"]
            if family["family_id"] == "mixed_p050"
        )
        corrected = family["corrected_policy_reference_counts"]
        self.assertEqual(corrected["rows_total"], 20776)
        self.assertEqual(corrected["paired_objects_total"], 10388)
        self.assertEqual(corrected["train_pairs_total"], 8932)
        self.assertEqual(
            corrected["paired_objects_total_by_source"]["visomaster_hints"]
            + corrected["paired_objects_total_by_source"]["visomaster_hints_teams"],
            682,
        )
        self.assertEqual(
            corrected["train_pairs_by_source"]["visomaster_hints"]
            + corrected["train_pairs_by_source"]["visomaster_hints_teams"],
            571,
        )

    def test_summary_mentions_key_labels_and_families(self):
        summary = self.tool.render_summary(self.data)
        self.assertIn("Training policy-aware today: False", summary)
        self.assertIn("[mixed_p050] old-semantics", summary)
        self.assertIn("[mixed_p070] old-semantics", summary)
        self.assertIn("[teamsonly_p050] old-semantics", summary)


if __name__ == "__main__":
    unittest.main()
