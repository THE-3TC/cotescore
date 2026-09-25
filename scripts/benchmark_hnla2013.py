#!/usr/bin/env python3
"""
Benchmark all models on the HNLA2013 dataset.

This script evaluates multiple models (DocLayout-YOLO, DoclingLayoutHeron,
PP-DocLayout-L) on the HNLA2013 evaluation set using SSU-tagged PAGE XML
ground truth and computes comparative COTe metrics.
"""

import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark all document layout models on HNLA2013")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to flat directory containing HNLA2013 PNG images",
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        required=True,
        help="Path to groundtruth_with_ssu/ directory containing PAGE XML files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hnla2013",
        help="Path to output directory (default: results/hnla2013)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "heron", "ppdoc-l"],
        choices=["yolo", "heron", "ppdoc-l", "ppdoc-m", "ppdoc-s"],
        help="Models to benchmark (default: yolo heron ppdoc)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument(
        "--image-ext",
        type=str,
        default="png",
        help="Image file extension (default: png)",
    )
    parser.add_argument(
        "--map-class-aware",
        action="store_true",
        help="If set, compute class-aware mAP (by default mAP ignores class).",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    groundtruth_path = Path(args.groundtruth)
    output_path = Path(args.output)

    if not dataset_path.exists():
        logger.error(f"Images path does not exist: {dataset_path}")
        sys.exit(1)

    if not groundtruth_path.exists():
        logger.error(f"Ground truth path does not exist: {groundtruth_path}")
        sys.exit(1)

    runner = BenchmarkRunner(
        dataset_path,
        output_path,
        image_ext=args.image_ext,
        dataset_name="hnla2013",
        groundtruth_path=groundtruth_path,
    )

    all_results = {"timestamp": datetime.now().isoformat(), "models": {}}

    models_to_run = []

    if "yolo" in args.models:
        from models.doclayout_yolo import DocLayoutYOLO

        models_to_run.append(
            ("DocLayout-YOLO", DocLayoutYOLO(device=args.device if args.device else "cpu"))
        )

    if "heron" in args.models:
        from models.docling_heron import DoclingLayoutHeron

        models_to_run.append(("DoclingLayoutHeron", DoclingLayoutHeron(device=args.device)))

    if "ppdoc-l" in args.models:
        from models.pp_doclayout import PPDocLayout

        models_to_run.append(
            ("PPDocLayout-L", PPDocLayout(device=args.device if args.device else "cpu"))
        )

    if "ppdoc-m" in args.models:
        from models.pp_doclayout import PPDocLayout

        models_to_run.append(
            (
                "PPDocLayout-M",
                PPDocLayout(
                    model_name="PP-DocLayout-M", device=args.device if args.device else "cpu"
                ),
            )
        )

    if "ppdoc-s" in args.models:
        from models.pp_doclayout import PPDocLayout

        models_to_run.append(
            (
                "PPDocLayout-S",
                PPDocLayout(
                    model_name="PP-DocLayout-S", device=args.device if args.device else "cpu"
                ),
            )
        )

    for model_name, model in models_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {model_name}...")
        logger.info(f"{'='*60}")

        try:
            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            results = runner.run_evaluation(
                model,
                map_ignore_class=(not args.map_class_aware),
                predictions_csv=output_path / f"{safe_name}_predictions.csv",
                model_label=model_name,
            )
            runner.print_summary(results)
            runner.save_results(results, filename=f"{safe_name}_results.json")

            all_results["models"][model_name] = results["metrics"]

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            import traceback

            traceback.print_exc()

    final_output = output_path / "benchmark_all_results.json"
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nCombined results saved to: {final_output}")

    metrics = [
        "map",
        "map_50",
        "map_75",
        "f1_50",
        "precision_50",
        "recall_50",
        "mean_iou",
        "coverage",
        "overlap",
        "trespass",
        "excess",
        "cot_score",
    ]
    metric_labels = {
        "map": "mAP (COCO)",
        "map_50": "mAP@50",
        "map_75": "mAP@75",
        "f1_50": "F1@50 (COCO)",
        "precision_50": "Precision@50",
        "recall_50": "Recall@50",
        "mean_iou": "Mean IoU",
        "coverage": "Coverage",
        "overlap": "Overlap",
        "trespass": "Trespass",
        "excess": "Excess",
        "cot_score": "COT Score",
    }

    run_models = list(all_results["models"].keys())
    col_w = 14
    total_w = 20 + (col_w + 3) * len(run_models)

    print("\n" + "=" * total_w)
    print("COMPARATIVE SUMMARY — HNLA2013")
    print("=" * total_w)
    header = f"{'Metric':<20}" + "".join(f" | {m:<{col_w}}" for m in run_models)
    print(header)
    print("-" * total_w)

    for metric in metrics:
        row = f"{metric_labels[metric]:<20}"
        for m in run_models:
            score = all_results["models"].get(m, {}).get(metric, "N/A")
            if metric == "cot_score":
                score_str = f"{score:+.4f}" if isinstance(score, float) else str(score)
            else:
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            row += f" | {score_str:<{col_w}}"
        print(row)
    print("=" * total_w)


if __name__ == "__main__":
    main()
