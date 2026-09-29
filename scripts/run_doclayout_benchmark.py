#!/usr/bin/env python3
"""
Run DocLayout-YOLO benchmark on NCSE dataset.

This script evaluates the DocLayout-YOLO model on the NCSE v2 test set
and computes coverage, overlap, and IoU metrics.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.runner import BenchmarkRunner
from models.doclayout_yolo import DocLayoutYOLO


def main():
    """Run the benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Benchmark DocLayout-YOLO on NCSE dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/ncse",
        help="Path to NCSE dataset directory (default: data/ncse)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to output directory for results (default: results)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ncse",
        choices=["ncse", "doclaynet"],
        help="Type of dataset to benchmark (default: ncse)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="juliozhao/DocLayout-YOLO-DocStructBench",
        help="Model name or path (default: juliozhao/DocLayout-YOLO-DocStructBench)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.2, help="Confidence threshold (default: 0.2)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1024, help="Image size for inference (default: 1024)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda:0, etc.) (default: cpu)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean_iou", "coverage", "overlap"],
        help="Metrics to compute (default: mean_iou coverage overlap)",
    )

    args = parser.parse_args()

    # Initialize model
    print("Initializing DocLayout-YOLO model...")
    model = DocLayoutYOLO(
        model_name=args.model, conf_threshold=args.conf, imgsz=args.imgsz, device=args.device
    )

    # Initialize benchmark runner
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    runner = BenchmarkRunner(dataset_path, output_path, dataset_name=args.dataset_name)

    # Run evaluation
    print("\n" + "=" * 60)
    print("Starting benchmark evaluation...")
    print("=" * 60)

    results = runner.run_evaluation(model, metrics=args.metrics)

    # Print and save results
    runner.print_summary(results)
    runner.save_results(results, filename="doclayout_yolo_results.json")


if __name__ == "__main__":
    main()
