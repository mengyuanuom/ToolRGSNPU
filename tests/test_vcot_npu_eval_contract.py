from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VCoTNpuEvaluationContractTest(unittest.TestCase):
    def test_runner_uses_exact_non_padding_evaluation_sampler(self):
        source = (ROOT / "toolrgs" / "engine" / "runner.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn("DistributedEvalSampler(", source)
        self.assertIn("num_replicas=cfg.world_size", source)
        self.assertIn("rank=cfg.rank", source)

    def test_validation_avoids_ddp_forward_and_globally_reduces_counts(self):
        source = (ROOT / "toolrgs" / "engine" / "val_loop.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('getattr(self.model, "module", self.model)', source)
        self.assertIn("evaluation_model(*inputs, **model_kwargs)", source)
        self.assertIn("dist.all_reduce(statistics, op=dist.ReduceOp.SUM)", source)
        self.assertIn("calculate_vcot_grasp_success(", source)


if __name__ == "__main__":
    unittest.main()
