import unittest

from toolrgs.engine.samplers import DistributedEvalSampler


class DistributedEvalSamplerTest(unittest.TestCase):
    def test_shards_cover_dataset_once_without_padding(self):
        dataset = list(range(10))
        samplers = [
            DistributedEvalSampler(dataset, num_replicas=3, rank=rank)
            for rank in range(3)
        ]
        shards = [list(sampler) for sampler in samplers]

        self.assertEqual(shards[0], [0, 3, 6, 9])
        self.assertEqual(shards[1], [1, 4, 7])
        self.assertEqual(shards[2], [2, 5, 8])
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(len(dataset))))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(sampler) for sampler in samplers], [4, 3, 3])

    def test_empty_rank_is_supported_for_tiny_evaluation_sets(self):
        sampler = DistributedEvalSampler([0, 1], num_replicas=4, rank=3)
        self.assertEqual(list(sampler), [])
        self.assertEqual(len(sampler), 0)

    def test_invalid_distributed_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            DistributedEvalSampler([0], num_replicas=0, rank=0)
        with self.assertRaises(ValueError):
            DistributedEvalSampler([0], num_replicas=2, rank=2)


if __name__ == "__main__":
    unittest.main()
