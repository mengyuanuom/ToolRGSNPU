import unittest

import torch.nn as nn

from toolrgs.models.base import BaseGraspModel, model_requires_depth


class RGBModel(nn.Module):
    pass


class DepthModel(BaseGraspModel):
    requires_depth = True

    def forward(self, *args, **kwargs):
        return None


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class DepthModelContractTest(unittest.TestCase):
    def test_rgb_model_does_not_request_depth(self):
        self.assertFalse(model_requires_depth(RGBModel()))

    def test_depth_contract_survives_parallel_wrapper(self):
        self.assertTrue(model_requires_depth(Wrapper(DepthModel())))


if __name__ == "__main__":
    unittest.main()
