import unittest

from toolrgs.registry import Registry, build_from_cfg
from toolrgs.structures import GraspModelResult, GraspOutput


class RegistryTest(unittest.TestCase):
    def test_decorator_alias_and_normalized_lookup(self):
        registry = Registry("test components")

        @registry.register_module(name="SmallThing", aliases=("small-thing",))
        class SmallThing:
            def __init__(self, value=1):
                self.value = value

        self.assertIs(registry.require("SMALLTHING"), SmallThing)
        self.assertIs(registry.require("small thing"), SmallThing)
        self.assertEqual(registry.build({"type": "small-thing", "value": 7}).value, 7)

    def test_default_args_do_not_override_explicit_config(self):
        registry = Registry("factories")
        registry.register_module(lambda value: value, name="identity")
        self.assertEqual(
            build_from_cfg(
                {"type": "identity", "value": 3},
                registry,
                default_args={"value": 9},
            ),
            3,
        )

    def test_duplicate_registration_is_rejected(self):
        registry = Registry("duplicates")
        registry.register_module(lambda: 1, name="same")
        with self.assertRaises(KeyError):
            registry.register_module(lambda: 2, name="same")

    def test_unknown_component_lists_available_names(self):
        registry = Registry("readable errors")
        registry.register_module(lambda: 1, name="known")
        with self.assertRaisesRegex(KeyError, "known"):
            registry.require("missing")


class GraspStructureTest(unittest.TestCase):
    def test_raw_prediction_tuple_becomes_named_output(self):
        result = GraspModelResult.from_legacy(("seg", "qua", "sin", "cos", "wid"))
        self.assertIsInstance(result.predictions, GraspOutput)
        self.assertEqual(result.predictions.quality, "qua")
        self.assertFalse(result.predictions.has_offset)

    def test_eval_and_training_contracts_round_trip(self):
        predictions = ("seg", "qua", "sin", "cos", "wid", "off")
        targets = ("seg_t", "qua_t", "sin_t", "cos_t", "wid_t", "off_t")
        eval_result = GraspModelResult.from_legacy((predictions, targets))
        self.assertTrue(eval_result.predictions.has_offset)
        self.assertEqual(eval_result.to_legacy(), (predictions, targets))

        training = (predictions, targets, "loss", {"m_qua": 0.5})
        train_result = GraspModelResult.from_legacy(training)
        self.assertTrue(train_result.is_training_result)
        self.assertEqual(train_result.to_legacy(), training)

    def test_incomplete_dense_output_is_rejected(self):
        with self.assertRaises(ValueError):
            GraspModelResult.from_legacy(("seg", "qua"))


class DeploymentRegistryTest(unittest.TestCase):
    def test_hardware_components_are_registered_without_opening_hardware(self):
        import deployment.audio  # noqa: F401
        import deployment.detector  # noqa: F401
        import deployment.sources  # noqa: F401
        from deployment.robot import build_robot_client
        from toolrgs.registry import AUDIO_INPUTS, CAMERAS, DETECTORS, ROBOT_CLIENTS

        self.assertIn("opencv", CAMERAS)
        self.assertIn("realsense", CAMERAS)
        self.assertIn("gstreamer", CAMERAS)
        self.assertIn("legacy_tcp", ROBOT_CLIENTS)
        self.assertIn("mmdetection", DETECTORS)
        self.assertIn("whisper", AUDIO_INPUTS)

        client = build_robot_client(
            {"type": "legacy_tcp", "host": "127.0.0.1", "port": 3000}
        )
        self.assertFalse(client.connected)


if __name__ == "__main__":
    unittest.main()
