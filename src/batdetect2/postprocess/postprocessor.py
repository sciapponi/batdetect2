from typing import List, Optional, Tuple, Union

import torch
from loguru import logger

from batdetect2.postprocess.config import (
    PostprocessConfig,
)
from batdetect2.postprocess.extraction import extract_detection_peaks
from batdetect2.postprocess.nms import NMS_KERNEL_SIZE, non_max_suppression
from batdetect2.postprocess.remapping import map_detection_to_clip
from batdetect2.typing import ModelOutput
from batdetect2.typing.postprocess import (
    ClipDetectionsTensor,
    PostprocessorProtocol,
)
from batdetect2.typing.preprocess import PreprocessorProtocol
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "build_postprocessor",
    "Postprocessor",
]


def build_postprocessor(
    preprocessor: PreprocessorProtocol,
    config: Optional[PostprocessConfig] = None,
    targets: Optional[TargetProtocol] = None,
) -> PostprocessorProtocol:
    """Factory function to build the standard postprocessor."""
    config = config or PostprocessConfig()
    logger.opt(lazy=True).debug(
        "Building postprocessor with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return Postprocessor(
        samplerate=preprocessor.output_samplerate,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        top_k_per_sec=config.top_k_per_sec,
        detection_threshold=config.detection_threshold,
        use_genus_prior=config.use_genus_prior,
        targets=targets,
    )


class Postprocessor(torch.nn.Module, PostprocessorProtocol):
    """Standard implementation of the postprocessing pipeline."""

    def __init__(
        self,
        samplerate: float,
        min_freq: float,
        max_freq: float,
        top_k_per_sec: int = 200,
        detection_threshold: float = 0.01,
        nms_kernel_size: Union[int, Tuple[int, int]] = NMS_KERNEL_SIZE,
        use_genus_prior: bool = False,
        targets: Optional[TargetProtocol] = None,
    ):
        """Initialize the Postprocessor."""
        super().__init__()

        self.output_samplerate = samplerate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.top_k_per_sec = top_k_per_sec
        self.detection_threshold = detection_threshold
        self.nms_kernel_size = nms_kernel_size
        self.use_genus_prior = use_genus_prior
        self.targets = targets

    def forward(
        self,
        output: ModelOutput,
        start_times: Optional[List[float]] = None,
    ) -> List[ClipDetectionsTensor]:
        detection_heatmap = non_max_suppression(
            output.detection_probs.detach(),
            kernel_size=self.nms_kernel_size,
        )

        width = output.detection_probs.shape[-1]
        duration = width / self.output_samplerate
        max_detections = int(self.top_k_per_sec * duration)
        
        # Prepare genus prior parameters if enabled
        genus_heatmap = None
        class_to_genus_idx = None
        if self.use_genus_prior and output.genus_probs is not None and self.targets is not None:
            genus_heatmap = output.genus_probs
            # Create mapping from class index to genus index
            # Assumes class_names are sorted and match the channel order in class_probs
            if hasattr(self.targets, 'class_to_genus') and hasattr(self.targets, 'class_names'):
                class_names = sorted(self.targets.class_names)
                class_to_genus_idx = {
                    idx: self.targets.class_to_genus.get(name)
                    for idx, name in enumerate(class_names)
                    if name in self.targets.class_to_genus
                }
        
        detections = extract_detection_peaks(
            detection_heatmap,
            size_heatmap=output.size_preds,
            feature_heatmap=output.features,
            classification_heatmap=output.class_probs,
            max_detections=max_detections,
            threshold=self.detection_threshold,
            genus_heatmap=genus_heatmap,
            class_to_genus_idx=class_to_genus_idx,
        )

        if start_times is None:
            start_times = [0 for _ in range(len(detections))]

        return [
            map_detection_to_clip(
                detection,
                start_time=0,
                end_time=duration,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )
            for detection in detections
        ]
