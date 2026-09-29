"""
PP-DocLayout-L model loader and inference.

This module provides a wrapper for the PaddlePaddle PP-DocLayout-L model
for document layout analysis, via the PaddleOCR LayoutDetection API.

Model card: https://huggingface.co/PaddlePaddle/PP-DocLayout-L
"""

from typing import Any, Dict, List
from pathlib import Path
import logging
from .loader import LayoutModel

logger = logging.getLogger(__name__)


class PPDocLayout(LayoutModel):
    """PP-DocLayout-L model for document layout analysis via PaddleOCR."""

    DEFAULT_MODEL = "PP-DocLayout-L"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        conf_threshold: float = 0.2,
        device: str = "cpu",
        enable_mkldnn: bool = False,
    ):
        """
        Initialize the PP-DocLayout-L model.

        Args:
            model_name: PaddleOCR model name (default: PP-DocLayout-L).
                        Other available variants include 'PP-DocLayout-M' and 'PP-DocLayout-S'.
            conf_threshold: Confidence threshold for filtering predictions (default: 0.2).
            device: Inference device – 'cpu' or 'gpu' (default: 'cpu').
                    Note: PaddleOCR uses 'gpu' rather than 'cuda'.
            enable_mkldnn: Enable oneDNN (MKLDNN) CPU acceleration (default: False).
                    Kept off because PP-DocLayout-L crashes under Paddle 3.0's PIR
                    executor with "ConvertPirAttribute2RuntimeAttribute not support
                    [pir::ArrayAttribute<pir::DoubleAttribute>]" in the oneDNN path.
        """
        super().__init__(model_name)
        self.conf_threshold = conf_threshold
        # PaddleOCR accepts 'gpu' not 'cuda:0', so normalise common variants
        self.device = self._normalise_device(device)
        self.enable_mkldnn = enable_mkldnn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_device(device: str) -> str:
        """Convert torch-style device strings to PaddleOCR style."""
        d = device.lower()
        if d.startswith("cuda"):
            return "gpu"
        return d  # 'cpu', 'gpu' pass through unchanged

    # ------------------------------------------------------------------
    # LayoutModel interface
    # ------------------------------------------------------------------

    def load(self):
        """Load the PP-DocLayout-L model via PaddleOCR."""
        try:
            from paddleocr import LayoutDetection
        except ImportError:
            raise ImportError(
                "paddleocr package not found. "
                "Install PaddlePaddle first (CPU): "
                "  pip install paddlepaddle==3.0.0 "
                "    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
                "Then install PaddleOCR:\n"
                "  pip install paddleocr"
            )

        logger.info(
            f"Loading PP-DocLayout model: {self.model_name} on device: {self.device} "
            f"(mkldnn={'on' if self.enable_mkldnn else 'off'})"
        )
        # PaddleOCR 3.x removed `use_gpu`; device is passed as a kwarg that the
        # underlying PaddleX pipeline forwards (accepts 'cpu', 'gpu', 'gpu:0', etc.).
        # enable_mkldnn defaults off: oneDNN crashes the PIR executor on this model.
        self.model = LayoutDetection(
            model_name=self.model_name,
            device=self.device,
            enable_mkldnn=self.enable_mkldnn,
        )
        logger.info("PP-DocLayout model loaded successfully.")

    def predict(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            image_path: Path to the input image.

        Returns:
            List of predicted regions with bounding boxes and labels in the format:
            [{'x': float, 'y': float, 'width': float, 'height': float,
              'class': str, 'confidence': float}, ...]
        """
        if self.model is None:
            self.load()

        try:
            output = self.model.predict(str(image_path), batch_size=1)
        except Exception as e:
            logger.error(f"Error running PP-DocLayout prediction on {image_path}: {e}")
            return []

        predictions: List[Dict[str, Any]] = []
        for res in output:
            raw = res.json if hasattr(res, "json") else {}
            boxes = raw.get("res", {}).get("boxes", [])
            predictions.extend(self._parse_boxes(boxes))

        return predictions

    def predict_batch(
        self, image_paths: List[Path], batch_size: int = 16
    ) -> List[List[Dict[str, Any]]]:
        """
        Run batched inference using PaddleOCR LayoutDetection's batch API.

        Args:
            image_paths: List of paths to input images.
            batch_size: Number of images per PaddleOCR batch call.

        Returns:
            List of prediction lists, one per image, preserving input order.
        """
        if self.model is None:
            self.load()

        all_predictions: List[List[Dict[str, Any]]] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            str_paths = [str(p) for p in batch_paths]

            # Errors propagate so BenchmarkRunner retries the chunk image by image.
            output = self.model.predict(str_paths, batch_size=len(batch_paths))
            for res in output:
                raw = res.json if hasattr(res, "json") else {}
                boxes = raw.get("res", {}).get("boxes", [])
                all_predictions.append(self._parse_boxes(boxes))

        return all_predictions

    def _parse_boxes(self, boxes: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse raw PP-DocLayout output boxes into the standard benchmark format.

        The model returns boxes with a 'coordinate' key in [x1, y1, x2, y2] (XYXY) format.
        """
        predictions = []
        for box in boxes:
            score = float(box.get("score", 0.0))
            if score < self.conf_threshold:
                continue

            coord = box.get("coordinate", [])
            if len(coord) != 4:
                logger.warning(f"Unexpected coordinate format: {coord!r} – skipping.")
                continue

            x1, y1, x2, y2 = coord
            predictions.append(
                {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "class": str(box.get("label", "unknown")),
                    "confidence": score,
                }
            )

        return predictions
