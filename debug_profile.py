"""Profiling script to identify bottlenecks in SEM V5.5 training.

Usage:
    uv run python debug_profile.py --config configs/rtx3060_max.yaml --device cuda

Runs exactly 5 training steps with torch.profiler enabled, then prints
a sorted table of the top 20 most time-consuming operators.
"""

import argparse
import logging
import os
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate

from sem.config import SEMConfig
from sem.training.trainer import SEMTrainer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Profile SEM V5.5 training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtx3060_max.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cpu/cuda/cuda:0)",
    )
    args = parser.parse_args()

    # Load config
    config = SEMConfig.from_yaml(args.config)

    # Override: run exactly 5 steps
    config.training.max_steps = 5
    config.training.wandb_enabled = False
    config.curriculum.enabled = False
    config.distillation.enabled = False

    logger.info(f"Profiling SEM V5.5 for exactly 5 steps on device={args.device}")
    logger.info(f"Config: {args.config}")

    # Create trainer
    trainer = SEMTrainer(
        config=config,
        device=args.device,
        dry_run=False,
    )

    # Profiler setup
    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    logger.info(f"Profiler activities: {activities}")
    logger.info("Starting profiling...")

    # Run training with profiler
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        trainer.train()

    # Export and analyze results
    logger.info("\n" + "=" * 80)
    logger.info("PROFILING RESULTS - Top 20 Most Time-Consuming Operators")
    logger.info("=" * 80)

    # Get key averages
    key_avg = prof.key_averages()

    # Sort by self_cpu_time_total (descending)
    sorted_by_cpu = sorted(key_avg, key=lambda x: x.self_cpu_time_total, reverse=True)[
        :20
    ]

    # Build table data
    table_data = []
    for i, stat in enumerate(sorted_by_cpu, 1):
        op_name = stat.key
        cpu_time_ms = stat.self_cpu_time_total / 1000.0  # Convert to ms
        cuda_time_ms = (
            stat.self_cuda_time_total / 1000.0
            if hasattr(stat, "self_cuda_time_total")
            else 0.0
        )
        calls = stat.count

        table_data.append(
            [
                i,
                op_name,
                f"{cpu_time_ms:.2f}",
                f"{cuda_time_ms:.2f}",
                calls,
            ]
        )

    headers = ["Rank", "Operator", "Self CPU (ms)", "Self CUDA (ms)", "Calls"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    # Also print by CUDA time if available
    if args.device.startswith("cuda"):
        logger.info("\n" + "=" * 80)
        logger.info("PROFILING RESULTS - Top 20 by Self CUDA Time")
        logger.info("=" * 80)

        sorted_by_cuda = sorted(
            key_avg,
            key=lambda x: x.self_cuda_time_total
            if hasattr(x, "self_cuda_time_total")
            else 0.0,
            reverse=True,
        )[:20]

        table_data_cuda = []
        for i, stat in enumerate(sorted_by_cuda, 1):
            op_name = stat.key
            cpu_time_ms = stat.self_cpu_time_total / 1000.0
            cuda_time_ms = (
                stat.self_cuda_time_total / 1000.0
                if hasattr(stat, "self_cuda_time_total")
                else 0.0
            )
            calls = stat.count

            table_data_cuda.append(
                [
                    i,
                    op_name,
                    f"{cpu_time_ms:.2f}",
                    f"{cuda_time_ms:.2f}",
                    calls,
                ]
            )

        print("\n" + tabulate(table_data_cuda, headers=headers, tablefmt="grid"))

    logger.info("\n" + "=" * 80)
    logger.info("Profiling complete. Check tables above for bottlenecks.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
