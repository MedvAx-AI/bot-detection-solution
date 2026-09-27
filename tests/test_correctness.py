"""Checks for two consequential risks: metric ties and future-event leakage."""
import unittest
import warnings
import pandas as pd
from bot_detection.metric import precision_at_recall
from bot_detection.features import build_features, pointer_features, device_features


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

    def test_metric_groups_ties_and_maximizes_precision(self):
        self.assertEqual(precision_at_recall([1, 1, 0, 0], [.5] * 4), .5)
        self.assertEqual(precision_at_recall([0, 0, 1, 1], [.5] * 4), .5)
        self.assertEqual(precision_at_recall([1, 0, 1, 1, 1], [5, 4, 3, 2, 1]), .8)

    def test_events_outside_window_cannot_change_features(self):
        meta = pd.DataFrame({"cookie_id": ["example"], "cookie_created_at": ["2026-04-01"],
                             "window_start_ts": ["2026-04-06"], "window_end_ts": ["2026-04-07"]})
        row = {"cookie_id": "example", "event_ts": pd.Timestamp("2026-04-06"),
               "eid": 100, "event_name": "search_results_view", "platform": "web",
               "user_agent": "Mozilla Chrome", "item_id": 123., "item_category": "electronics",
               "item_location": "moscow", "seller_type": "private", "search_query": "phone",
               "search_page": 1., "pointer_x": 100., "pointer_y": 200.}
        valid = pd.DataFrame([row, {**row, "event_ts": pd.Timestamp("2026-04-06 23:59:59"),
                                    "eid": 200, "event_name": "item_view", "pointer_x": 300.}])
        outside = pd.DataFrame([{**row, "event_ts": pd.Timestamp(time), "pointer_x": 99999.}
                                for time in ["2026-04-05 23:59:59", "2026-04-07", "2026-04-08"]])
        for build in [build_features, pointer_features, device_features]:
            pd.testing.assert_frame_equal(build(meta, valid), build(meta, pd.concat([valid, outside], ignore_index=True)))
        self.assertEqual(int(build_features(meta, valid).n_events.iloc[0]), 2)


if __name__ == "__main__":
    unittest.main()
