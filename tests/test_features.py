import unittest
import numpy as np
from twitter_time_hawkes_rumor import TreeData, hawkes_temporal_features, bfs_depth

class FeatureTests(unittest.TestCase):
    def test_empty_and_nonempty_have_same_dimension(self):
        def tree(delays):
            return TreeData("event", "root", 0, delays, [], {}, {}, {})
        empty = hawkes_temporal_features(tree([]), .08, 256)
        populated = hawkes_temporal_features(tree([0, 1, 5]), .08, 256)
        self.assertEqual(empty.shape, (31,))
        self.assertEqual(empty.shape, populated.shape)
        self.assertTrue(np.isfinite(populated).all())
    def test_bfs_cycle_terminates(self):
        self.assertEqual(bfs_depth({"a": ["b"], "b": ["a"]}, "a"), (1, .5))

if __name__ == "__main__":
    unittest.main()
