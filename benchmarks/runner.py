"""Benchmark runner for evaluating models on the NCSE dataset."""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from tqdm import tqdm

# torch is only needed for CUDA synchronization in latency measurement.
# It is optional so that PaddlePaddle-only benchmarks work without PyTorch.
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from cotescore.dataset import (
    NCSEDataset,
    HNLA2013Dataset,
    DocLayNetDataset,
    SpiritualistDataset,
    HierTextDataset,
)
from cotescore.adapters import compute_canvas, boxes_to_gt_ssu_map, boxes_to_pred_masks
from cotescore.types import MaskInstance
from cotescore.layout import (
    coverage,
    overlap,
    trespass,
    excess,
    cote_score as cot_score,
    mean_iou,
)
from cotescore.map_metric import MAPMetric
from PIL import Image

logger = logging.getLogger(__name__)


EVAL_MAX_DIM = 2000

# Same columns as scripts/export_predictions.py, so either output can be analysed the same way.
PREDICTION_CSV_COLUMNS = [
    "filename", "image_path", "image_width", "image_height", "model", "source",
    "x", "y", "width", "height", "class", "confidence", "ssu_id",
]

# Metrics computed by MAPMetric (pycocotools COCOeval) over the whole set rather than per image.
COCO_METRICS = ("map", "f1_50")


def _mask_instances_to_canvas(
    predictions: List[MaskInstance],
    canvas_w: int,
    canvas_h: int,
) -> List[np.ndarray]:
    """Resize MaskInstance masks (image-resolution) to canvas resolution."""
    masks = []
    for inst in predictions:
        pil_mask = Image.fromarray(inst.mask.astype(np.uint8) * 255)
        pil_mask = pil_mask.resize((canvas_w, canvas_h), Image.NEAREST)
        masks.append(np.array(pil_mask) > 0)
    return masks


def _prediction_rows(img_result: dict, model_label: str) -> List[dict]:
    """GT and predicted boxes of one image as rows of PREDICTION_CSV_COLUMNS."""
    base = {
        "filename": img_result["filename"],
        "image_path": img_result["image_path"],
        "image_width": img_result["image_width"],
        "image_height": img_result["image_height"],
        "model": model_label,
    }
    rows = []
    for source, boxes in (("gt", img_result["ground_truth"]), ("pred", img_result["predictions"])):
        for b in boxes:
            if not isinstance(b, dict):
                continue  # mask predictions have no box to record
            rows.append(
                {
                    **base,
                    "source": source,
                    "x": b["x"],
                    "y": b["y"],
                    "width": b["width"],
                    "height": b["height"],
                    "class": b["class"],
                    "confidence": b.get("confidence", 1.0 if source == "gt" else None),
                    "ssu_id": b.get("ssu_id") if source == "gt" else None,
                }
            )
    return rows


def _compute_image_metrics(
    sample: dict,
    predictions: List[Dict],
    metrics: List[str],
) -> dict:
    """
    Compute all per-image metrics. Pure numpy — thread-safe.

    Returns dict with keys: filename, predictions, ground_truth, image_metrics.
    """
    image_path = Path(sample["image_path"])
    ground_truth = sample["annotations"]

    try:
        with Image.open(image_path) as img:
            image_width, image_height = img.size
    except Exception as e:
        logger.warning(f"Failed to get image dimensions for {sample['filename']}: {e}")
        image_width, image_height = 1000, 1000

    canvas_w, canvas_h = compute_canvas(image_width, image_height, EVAL_MAX_DIM)
    gt_ssu_map = boxes_to_gt_ssu_map(ground_truth, image_width, image_height, canvas_w, canvas_h)

    is_mask_preds = bool(predictions) and isinstance(predictions[0], MaskInstance)
    if is_mask_preds:
        pred_masks = _mask_instances_to_canvas(predictions, canvas_w, canvas_h)
    else:
        pred_masks = boxes_to_pred_masks(predictions, image_width, image_height, canvas_w, canvas_h)

    image_metrics = {}
    for metric_name in metrics:
        if metric_name in COCO_METRICS:
            continue  # computed over the whole set by MAPMetric
        elif metric_name == "mean_iou" and is_mask_preds:
            score = float("nan")  # not defined for mask predictions
        elif metric_name == "mean_iou":
            score = mean_iou(predictions, ground_truth)
        elif metric_name == "coverage":
            score = coverage(gt_ssu_map, pred_masks)
        elif metric_name == "overlap":
            score = overlap(gt_ssu_map, pred_masks)
        elif metric_name == "trespass":
            score = trespass(gt_ssu_map, pred_masks)
        elif metric_name == "excess":
            score = excess(gt_ssu_map, pred_masks)
        elif metric_name == "cot_score":
            score = cot_score(gt_ssu_map, pred_masks)[0]
        else:
            logger.warning(f"Unknown per-image metric: {metric_name}")
            score = 0.0

        image_metrics[metric_name] = score

    return {
        "filename": sample["filename"],
        "image_path": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "image_metrics": image_metrics,
    }


class BenchmarkRunner:
    """Orchestrates model evaluation on layout datasets."""

    def __init__(
        self,
        dataset_path: Path,
        output_path: Path,
        csv_filename: str = None,
        images_subdir: str = None,
        image_ext: str = "png",
        dataset_name: str = "ncse",
        groundtruth_path: Path = None,
        split: str = "test",
    ):
        """
        Initialize the benchmark runner.

        Args:
            dataset_path: Path to dataset (or images directory for HNLA2013).
                For DocLayNet, may be None to trigger HuggingFace download.
            output_path: Path where results will be saved
            csv_filename: Name of the annotations CSV file (for NCSE)
            images_subdir: Name of the images subdirectory (for NCSE)
            image_ext: Image file extension to glob for
            dataset_name: Type of dataset to load ('ncse', 'doclaynet', or 'hnla2013')
            groundtruth_path: Path to ground truth directory (for HNLA2013)
            split: Dataset split for DocLayNet (default: 'test')
        """
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.csv_filename = csv_filename
        self.images_subdir = images_subdir
        self.image_ext = image_ext
        self.dataset_name = dataset_name.lower()
        self.groundtruth_path = Path(groundtruth_path) if groundtruth_path else None
        self.split = split
        self._dataset = None  # cached dataset, loaded once across model runs

    def measure_latency(
        self, model, sample_image_path: Path, warmup: int = 10, repeats: int = 50
    ) -> Dict[str, float]:
        """
        Measure inference latency for a model.

        Args:
            model: Loaded model instance
            sample_image_path: Path to a sample image
            warmup: Number of warmup iterations
            repeats: Number of measurement iterations

        Returns:
            Dictionary with mean and std latency in milliseconds
        """
        device = getattr(model, "device", "unknown")
        logger.info(f"Measuring latency on {device}...")

        def _cuda_sync():
            """Synchronize CUDA if torch is available and model is on GPU."""
            if TORCH_AVAILABLE and torch.cuda.is_available() and "cuda" in str(device):
                torch.cuda.synchronize()

        # Warmup
        for _ in range(warmup):
            _ = model.predict(sample_image_path)

        _cuda_sync()

        # Timing
        latencies = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model.predict(sample_image_path)
            _cuda_sync()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }

    def run_evaluation(
        self,
        model,
        metrics: List[str] = None,
        *,
        map_ignore_class: bool = True,
        batch_size: int = 16,
        predictions_csv: Optional[Path] = None,
        model_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation for a model using specified metrics.

        Inference is batched (predict_batch) and per-image metric computation
        runs in parallel on all CPU cores via ThreadPoolExecutor.

        Args:
            model: Model instance to evaluate
            metrics: List of metric names (default: all)
            map_ignore_class: If True, collapse all classes to 'object' for mAP
                and F1 (both use COCO matching, which is class-aware)
            batch_size: Number of images per GPU inference batch (default: 16)
            predictions_csv: If given, write every GT and predicted box to this
                CSV (columns ``PREDICTION_CSV_COLUMNS``) so metrics can be
                recomputed later without re-running inference.
            model_label: Value for the CSV ``model`` column (default: model.model_name)

        Returns:
            Dictionary containing evaluation results
        """
        if metrics is None:
            metrics = [
                "mean_iou",
                "f1_50",
                "coverage",
                "overlap",
                "trespass",
                "excess",
                "cot_score",
                "map",
            ]

        if self._dataset is None:
            logger.info(f"Loading {self.dataset_name.upper()} dataset from {self.dataset_path}")

            if self.dataset_name == "ncse":
                self._dataset = NCSEDataset(
                    self.dataset_path,
                    split="test",
                    csv_filename=self.csv_filename,
                    images_subdir=self.images_subdir,
                    image_ext=self.image_ext,
                )
            elif self.dataset_name == "doclaynet":
                self._dataset = DocLayNetDataset(self.dataset_path, split=self.split)
            elif self.dataset_name == "hnla2013":
                if self.groundtruth_path is None:
                    raise ValueError("groundtruth_path must be provided for hnla2013 dataset")
                self._dataset = HNLA2013Dataset(
                    images_path=self.dataset_path,
                    groundtruth_path=self.groundtruth_path,
                    image_ext=self.image_ext,
                )
            elif self.dataset_name == "spiritualist":
                if self.groundtruth_path is None:
                    raise ValueError("groundtruth_path must be provided for spiritualist dataset")
                self._dataset = SpiritualistDataset(
                    images_path=self.dataset_path,
                    groundtruth_path=self.groundtruth_path,
                    image_ext=self.image_ext,
                )
            elif self.dataset_name == "hiertext":
                if self.groundtruth_path is None:
                    raise ValueError("groundtruth_path must be provided for hiertext dataset")
                self._dataset = HierTextDataset(
                    images_path=self.dataset_path,
                    groundtruth_path=self.groundtruth_path,
                    image_ext=self.image_ext,
                )
            else:
                raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

            self._dataset.load()

        dataset = self._dataset
        n = len(dataset)
        logger.info(f"Dataset loaded: {n} images")

        if model.model is None:
            model.load()

        # --- Phase 1: Collect all samples (fast — no disk I/O) ---
        samples = [dataset[i] for i in range(n)]
        all_image_paths = [Path(s["image_path"]) for s in samples]

        # --- Phases 2+3: Pipelined inference and metrics ---
        # Metric tasks for batch K are submitted immediately after inference completes,
        # so CPU metric computation overlaps with GPU inference for batch K+1.
        map_metric = MAPMetric() if any(m in metrics for m in COCO_METRICS) else None

        results = {
            "model": model.model_name,
            "dataset": f"{self.dataset_name.upper()}_test",
            "num_images": n,
            "metrics": {},
            "per_image_results": [],
            "classes": {},
        }
        metric_totals = {m: 0.0 for m in metrics if m not in COCO_METRICS}

        ordered_futures: List[Future] = []
        logger.info(
            f"Running pipelined inference + metrics (batch_size={batch_size}, workers={os.cpu_count()})..."
        )

        csv_file = csv_writer = None
        if predictions_csv is not None:
            predictions_csv = Path(predictions_csv)
            predictions_csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(predictions_csv, "w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_CSV_COLUMNS)
            csv_writer.writeheader()
        model_label = model_label or model.model_name

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for start in tqdm(range(0, n, batch_size), desc="Inference"):
                chunk_paths = all_image_paths[start : start + batch_size]
                chunk_samples = samples[start : start + batch_size]

                # GPU inference for this chunk
                try:
                    chunk_preds = model.predict_batch(chunk_paths, batch_size=len(chunk_paths))
                except Exception as e:
                    logger.error(f"predict_batch failed on chunk {start}: {e}")
                    chunk_preds = []
                    for path in chunk_paths:
                        try:
                            chunk_preds.append(model.predict(path))
                        except Exception as ex:
                            logger.error(f"Error predicting {path}: {ex}")
                            chunk_preds.append([])

                # Immediately submit metrics to CPU threads — runs while GPU does next chunk
                for sample, preds in zip(chunk_samples, chunk_preds):
                    ordered_futures.append(
                        executor.submit(_compute_image_metrics, sample, preds, metrics)
                    )

            # Collect in submission order to keep MAP updates serial
            for future in tqdm(ordered_futures, desc="Metrics"):
                img_result = future.result()

                # MAP update must stay serial and in-order (not thread-safe)
                if map_metric:
                    preds = img_result["predictions"]
                    gt = img_result["ground_truth"]
                    if map_ignore_class:
                        preds = [{**p, "class": "object"} for p in preds]
                        gt = [{**g, "class": "object"} for g in gt]
                    map_metric.update(preds, gt)

                if csv_writer:
                    csv_writer.writerows(_prediction_rows(img_result, model_label))

                for metric_name, score in img_result["image_metrics"].items():
                    if metric_name in metric_totals:
                        metric_totals[metric_name] += score

                results["per_image_results"].append(
                    {"filename": img_result["filename"], "metrics": img_result["image_metrics"]}
                )

        if csv_file:
            csv_file.close()
            logger.info(f"Predictions saved to: {predictions_csv}")

        # Calculate average metrics
        for metric_name in metric_totals:
            results["metrics"][metric_name] = metric_totals[metric_name] / n

        # Compute global mAP
        if map_metric:
            logger.info("Computing global mAP...")
            map_scores = map_metric.compute()
            if "map" in metrics:
                for key in ("map", "map_50", "map_75"):
                    results["metrics"][key] = map_scores[key]
                results["classes"] = map_scores["classes"]
            if "f1_50" in metrics:
                # f1_50 pools TP/FP/FN over the set, so every region counts equally (COCO
                # convention). f1_50_page_mean averages per-page F1, so every page counts
                # equally, matching how COTe and the other area metrics are averaged.
                for key in ("f1_50", "precision_50", "recall_50"):
                    results["metrics"][key] = map_scores[key]
                results["metrics"]["f1_50_page_mean"] = float(np.mean(map_scores["per_image_f1_50"]))
                for img, score in zip(results["per_image_results"], map_scores["per_image_f1_50"]):
                    img["metrics"]["f1_50"] = score

        return results

    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save evaluation results to disk."""
        output_file = self.output_path / filename
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {results['model']}")
        print(f"Images: {results['num_images']}")
        print("\nOverall Metrics:")
        print("-" * 60)

        metrics = results["metrics"]

        # Priority Print
        if "map" in metrics:
            print(f"  mAP (COCO)     : {metrics['map']:.4f}")
            print(f"  mAP@50         : {metrics['map_50']:.4f}")
            print(f"  mAP@75         : {metrics['map_75']:.4f}")
            print("-" * 30)

        for name in ["mean_iou", "f1_50", "f1_50_page_mean", "precision_50", "recall_50", "coverage", "overlap", "trespass"]:
            if name in metrics:
                print(f"  {name.upper():15s}: {metrics[name]:.4f}")

        # Print COT score separately (can be negative)
        if "cot_score" in metrics:
            print("-" * 30)
            print(f"  {'COT SCORE':15s}: {metrics['cot_score']:+.4f}")

        if "mean_latency_ms" in metrics:
            print("-" * 30)
            print(
                f"  Latency (ms)   : {metrics['mean_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f}"
            )

        if results.get("classes"):
            print("\nPer-Class AP:")
            print("-" * 30)
            # Sort by class name
            for cls_name, score in sorted(results["classes"].items()):
                print(f"  {cls_name:15s}: {score:.4f}")

        print("=" * 60)
