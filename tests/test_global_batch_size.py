import unittest

from toolrgs.engine.batch import per_process_batch_size


class GlobalBatchSizeTest(unittest.TestCase):
    def test_global_batch_is_split_across_processes(self):
        self.assertEqual(per_process_batch_size(24, 8, "batch_size"), 3)
        self.assertEqual(per_process_batch_size(24, 1, "batch_size"), 24)

    def test_global_batch_must_be_divisible_by_world_size(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            per_process_batch_size(25, 8, "batch_size")

    def test_global_batch_and_world_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            per_process_batch_size(0, 8, "batch_size")
        with self.assertRaisesRegex(ValueError, "world_size must be positive"):
            per_process_batch_size(24, 0, "batch_size")


if __name__ == "__main__":
    unittest.main()
