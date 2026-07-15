"""Compatibility CLI for the MMEngine-style ToolRGSNPU runner."""

import argparse

import utils.config as config
from toolrgs.engine.runner import build_runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train ToolRGSNPU")
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument(
        "--npu", type=int, default=0, help="NPU index for single-process runs"
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.npu = cli.npu
    return cfg


def main():
    runner = build_runner(parse_args())
    runner.train()


if __name__ == "__main__":
    main()
