"""Run one OCID-VLG optimizer step for a ToolRGS experiment config."""

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.config as config  # noqa: E402
from toolrgs.engine.runner import build_runner  # noqa: E402
from toolrgs.runtime import get_torch_npu, set_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--npu", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.npu = cli.npu
    cfg.gpu = cli.npu
    cfg.batch_size = 1
    cfg.batch_size_val = 1
    cfg.workers = 0
    cfg.workers_val = 0
    cfg.pin_memory = False
    cfg.print_freq = 1
    cfg.exp_name = f"smoke_{cfg.exp_name}"
    if cli.input_size is not None:
        cfg.input_size = cli.input_size

    started = time.time()
    set_device(cli.npu)
    torch_npu = get_torch_npu()
    torch_npu.npu.reset_peak_memory_stats()
    runner = build_runner(cfg).setup()
    batch = next(iter(runner.train_loader))
    runner.train_loop.dataloader = [batch]
    logs = runner.train_loop.run_epoch(1)
    torch_npu.npu.synchronize()
    result = {
        "architecture": str(cfg.architecture),
        "config": cli.config,
        "input_size": int(cfg.input_size),
        "loss": float(logs["loss"]),
        "peak_npu_mib": round(torch_npu.npu.max_memory_allocated() / 2**20, 1),
        "seconds": round(time.time() - started, 2),
    }
    print("SMOKE_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
