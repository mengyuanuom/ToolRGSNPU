"""Launch the ToolRGS real-world grasp GUI."""

import argparse

from deployment.config import load_deployment_config
from deployment.gui import run_gui


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/deployment/lab.example.yaml", help="Deployment YAML"
    )
    parser.add_argument(
        "--allow-robot",
        action="store_true",
        help="Permit the GUI to connect to a robot receiver when robot.enabled is true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_deployment_config(args.config)
    return run_gui(config, allow_robot=args.allow_robot)


if __name__ == "__main__":
    raise SystemExit(main())
