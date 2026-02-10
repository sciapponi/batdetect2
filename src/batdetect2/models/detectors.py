"""Assembles the complete BatDetect2 Detection Model.

This module defines the concrete `Detector` class, which implements the
`DetectionModel` interface defined in `.types`. It combines a feature
extraction backbone with specific prediction heads to create the end-to-end
neural network used for detecting bat calls, predicting their size, and
classifying them.

The primary components are:
- `Detector`: The `torch.nn.Module` subclass representing the complete model.

This module focuses purely on the neural network architecture definition. The
logic for preprocessing inputs and postprocessing/decoding outputs resides in
the `batdetect2.preprocess` and `batdetect2.postprocess` packages, respectively.
"""

from typing import Optional

import torch
from loguru import logger

from batdetect2.models.backbones import BackboneConfig, build_backbone
from batdetect2.models.heads import BBoxHead, ClassifierHead, GenusClassifierHead
from batdetect2.typing.models import BackboneModel, DetectionModel, ModelOutput

__all__ = [
    "Detector",
    "build_detector",
]


class Detector(DetectionModel):
    """Concrete implementation of the BatDetect2 Detection Model.

    Assembles a complete detection and classification model by combining a
    feature extraction backbone network with specific prediction heads for
    detection probability, bounding box size regression, and class
    probabilities.

    Attributes
    ----------
    backbone : BackboneModel
        The feature extraction backbone network module.
    num_classes : int
        The number of specific target classes the model predicts (derived from
        the `classifier_head`).
    classifier_head : ClassifierHead
        The prediction head responsible for generating class probabilities.
    bbox_head : BBoxHead
        The prediction head responsible for generating bounding box size
        predictions.
    """

    backbone: BackboneModel
    genus_head: Optional[GenusClassifierHead]

    def __init__(
        self,
        backbone: BackboneModel,
        classifier_head: ClassifierHead,
        bbox_head: BBoxHead,
        genus_head: Optional[GenusClassifierHead] = None,
    ):
        """Initialize the Detector model.

        Note: Instances are typically created using the `build_detector`
        factory function.

        Parameters
        ----------
        backbone : BackboneModel
            An initialized feature extraction backbone module (e.g., built by
            `build_backbone` from the `.backbone` module).
        classifier_head : ClassifierHead
            An initialized classification head module. The number of classes
            is inferred from this head.
        bbox_head : BBoxHead
            An initialized bounding box size prediction head module.

        Raises
        ------
        TypeError
            If the provided modules are not of the expected types.
        """
        super().__init__()

        self.backbone = backbone
        self.num_classes = classifier_head.num_classes
        self.classifier_head = classifier_head
        self.bbox_head = bbox_head
        self.genus_head = genus_head

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        """Perform the forward pass of the complete detection model.

        Processes the input spectrogram through the backbone to extract
        features, then passes these features through the separate prediction
        heads to generate detection probabilities, class probabilities, and
        size predictions.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, typically with shape
            `(batch_size, input_channels, frequency_bins, time_bins)`. The
            shape must be compatible with the `self.backbone` input
            requirements.

        Returns
        -------
        ModelOutput
            A NamedTuple containing the four output tensors:
            - `detection_probs`: Detection probability heatmap `(B, 1, H, W)`.
            - `size_preds`: Predicted scaled size dimensions `(B, 2, H, W)`.
            - `class_probs`: Class probabilities (excluding background)
              `(B, num_classes, H, W)`.
            - `features`: Output feature map from the backbone
              `(B, C_out, H, W)`.
        """
        features = self.backbone(spec)
        classification = self.classifier_head(features)
        detection = classification.sum(dim=1, keepdim=True)
        size_preds = self.bbox_head(features)
        
        genus_probs = None
        if self.genus_head is not None:
            genus_probs = self.genus_head(features)
        
        return ModelOutput(
            detection_probs=detection,
            size_preds=size_preds,
            class_probs=classification,
            features=features,
            genus_probs=genus_probs,
        )


def build_detector(
    num_classes: int,
    config: Optional[BackboneConfig] = None,
    num_genera: Optional[int] = None,
) -> DetectionModel:
    """Build the complete BatDetect2 detection model.

    Parameters
    ----------
    num_classes : int
        The number of specific target classes the model should predict
        (required for the `ClassifierHead`). Must be positive.
    config : BackboneConfig, optional
        Configuration object specifying the architecture of the backbone
        (encoder, bottleneck, decoder). If None, default configurations defined
        within the respective builder functions (`build_encoder`, etc.) will be
        used to construct a default backbone architecture.

    Returns
    -------
    DetectionModel
        An initialized `Detector` model instance.

    Raises
    ------
    ValueError
        If `num_classes` is not positive, or if errors occur during the
        construction of the backbone or detector components (e.g., incompatible
        configurations, invalid parameters).
    """
    config = config or BackboneConfig()

    logger.opt(lazy=True).debug(
        "Building model with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    backbone = build_backbone(config=config)
    classifier_head = ClassifierHead(
        num_classes=num_classes,
        in_channels=backbone.out_channels,
    )
    bbox_head = BBoxHead(
        in_channels=backbone.out_channels,
    )
    
    genus_head = None
    if num_genera is not None and num_genera > 0:
        genus_head = GenusClassifierHead(
            num_genera=num_genera,
            in_channels=backbone.out_channels,
        )
        logger.info(f"Building model with genus classification head ({num_genera} genera)")
    
    detector = Detector(
        backbone=backbone,
        classifier_head=classifier_head,
        bbox_head=bbox_head,
        genus_head=genus_head,
    )
    
    # Print parameter counts breakdown
    encoder_params = sum(p.numel() for p in backbone.encoder.parameters())
    bottleneck_params = sum(p.numel() for p in backbone.bottleneck.parameters())
    decoder_params = sum(p.numel() for p in backbone.decoder.parameters())
    head_params = sum(p.numel() for p in classifier_head.parameters()) + sum(p.numel() for p in bbox_head.parameters())
    total_params = sum(p.numel() for p in detector.parameters())
    
    logger.info(f"Model parameter counts:")
    logger.info(f"  Encoder:    {encoder_params:>10,} params")
    
    # Check if bottleneck contains VQ layers
    bottleneck_info = f"  Bottleneck: {bottleneck_params:>10,} params"
    for layer in backbone.bottleneck.layers:
        layer_type = type(layer).__name__
        if 'VectorQuantizer' in layer_type:
            # Get codebook size from the layer
            codebook_size = getattr(layer, 'codebook_size', 'unknown')
            bottleneck_info += f" (VQ codebook: {codebook_size})"
            break
    logger.info(bottleneck_info)
    
    logger.info(f"  Decoder:    {decoder_params:>10,} params")
    logger.info(f"  Heads:      {head_params:>10,} params")
    logger.info(f"  Total:      {total_params:>10,} params")
    
    return detector
