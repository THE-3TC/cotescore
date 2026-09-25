"""Tests for the benchmarking framework."""

import pytest
from pathlib import Path

try:
    from benchmarks.runner import BenchmarkRunner

    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="benchmarks dependencies not installed")
class TestBenchmarkRunner:
    """Tests for the BenchmarkRunner."""

    def test_runner_initialization(self):
        """Test benchmark runner initialization."""
        dataset_path = Path("fake/path")
        output_path = Path("fake/output")
        runner = BenchmarkRunner(dataset_path, output_path, dataset_name="ncse")

        assert runner.dataset_path == dataset_path
        assert runner.output_path == output_path
        assert runner.dataset_name == "ncse"

    def test_runner_initialization_doclaynet(self):
        """Test benchmark runner initialization with doclaynet."""
        dataset_path = Path("fake/path")
        output_path = Path("fake/output")
        runner = BenchmarkRunner(dataset_path, output_path, dataset_name="doclaynet")

        assert runner.dataset_path == dataset_path
        assert runner.output_path == output_path
        assert runner.dataset_name == "doclaynet"

    def test_evaluation_pipeline(self):
        """Test evaluation pipeline."""
        # TODO: Implement test
        pass

    def test_results_saving(self):
        """Test results saving."""
        # TODO: Implement test
        pass


@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="benchmarks dependencies not installed")
class TestRunEvaluation:
    """End-to-end run of the runner with a stub model and dataset."""

    GT = [
        [{"x": 10, "y": 10, "width": 40, "height": 20, "class": "text", "ssu_id": 1},
         {"x": 10, "y": 50, "width": 40, "height": 20, "class": "text", "ssu_id": 2}],
        [{"x": 5, "y": 5, "width": 50, "height": 50, "class": "text", "ssu_id": 1}],
    ]
    PREDS = [
        [{"x": 10, "y": 10, "width": 40, "height": 20, "class": "Text", "confidence": 0.9},
         {"x": 70, "y": 70, "width": 10, "height": 10, "class": "Text", "confidence": 0.8}],
        [{"x": 5, "y": 5, "width": 50, "height": 50, "class": "Text", "confidence": 0.95}],
    ]

    def _run(self, tmp_path):
        from PIL import Image

        samples = []
        for i, gt in enumerate(self.GT):
            path = tmp_path / f"page_{i}.png"
            Image.new("RGB", (100, 100), "white").save(path)
            samples.append({"image_path": str(path), "annotations": gt, "filename": path.name})

        class StubDataset(list):
            def load(self):
                pass

        class StubModel:
            model_name = "stub/model"
            model = object()
            preds = {s["image_path"]: p for s, p in zip(samples, self.PREDS)}

            def predict_batch(self, paths, batch_size):
                return [self.preds[str(p)] for p in paths]

        runner = BenchmarkRunner(tmp_path, tmp_path / "out", dataset_name="ncse")
        runner._dataset = StubDataset(samples)
        csv_path = tmp_path / "out" / "stub_predictions.csv"
        results = runner.run_evaluation(
            StubModel(), predictions_csv=csv_path, model_label="Stub", batch_size=1
        )
        return results, csv_path

    def test_f1_is_coco_matched_and_pooled(self, tmp_path):
        results, _ = self._run(tmp_path)
        # TP=2, FP=1, FN=1 pooled over both images (classes collapsed to 'object').
        assert results["metrics"]["f1_50"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
        assert results["metrics"]["precision_50"] == pytest.approx(2 / 3)
        assert results["metrics"]["recall_50"] == pytest.approx(2 / 3)
        per_image = [r["metrics"]["f1_50"] for r in results["per_image_results"]]
        assert per_image == pytest.approx([0.5, 1.0])
        assert results["metrics"]["f1_50_page_mean"] == pytest.approx(0.75)

    def test_predictions_csv_reproduces_metrics(self, tmp_path):
        import pandas as pd
        from cotescore.map_metric import MAPMetric

        results, csv_path = self._run(tmp_path)
        df = pd.read_csv(csv_path)
        assert (df["model"] == "Stub").all()
        assert (df["source"] == "gt").sum() == 3 and (df["source"] == "pred").sum() == 3

        metric = MAPMetric()
        for _, g in df.groupby("filename", sort=False):
            boxes = lambda src: [
                {"x": r.x, "y": r.y, "width": r.width, "height": r.height,
                 "confidence": r.confidence, "class": "object"}
                for r in g[g.source == src].itertuples()
            ]
            metric.update(boxes("pred"), boxes("gt"))
        rescored = metric.compute()
        for key in ("map", "f1_50"):
            assert rescored[key] == pytest.approx(results["metrics"][key])
