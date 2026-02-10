from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Annotated,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from pydantic import Field
from sklearn import metrics
from soundevent import data

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.metrics.common import (
    average_precision,
    compute_precision_recall,
)
from batdetect2.typing import RawPrediction, TargetProtocol

__all__ = [
    "ClassificationMetric",
    "ClassificationMetricConfig",
    "build_classification_metric",
    "compute_precision_recall_curves",
]


@dataclass
class MatchEval:
    clip: data.Clip
    gt: Optional[data.SoundEventAnnotation]
    pred: Optional[RawPrediction]

    is_prediction: bool
    is_ground_truth: bool
    is_generic: bool
    true_class: Optional[str]
    score: float


@dataclass
class ClipEval:
    clip: data.Clip
    matches: Mapping[str, List[MatchEval]]


ClassificationMetric = Callable[[Sequence[ClipEval]], Dict[str, float]]


classification_metrics: Registry[ClassificationMetric, [TargetProtocol]] = (
    Registry("classification_metric")
)


class BaseClassificationConfig(BaseConfig):
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class BaseClassificationMetric:
    def __init__(
        self,
        targets: TargetProtocol,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.targets = targets
        self.include = include
        self.exclude = exclude

    def include_class(self, class_name: str) -> bool:
        if self.include is not None:
            return class_name in self.include

        if self.exclude is not None:
            return class_name not in self.exclude

        return True


class ClassificationAveragePrecisionConfig(BaseClassificationConfig):
    name: Literal["average_precision"] = "average_precision"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True
    label: str = "average_precision"


class ClassificationAveragePrecision(BaseClassificationMetric):
    def __init__(
        self,
        targets: TargetProtocol,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        label: str = "average_precision",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(include=include, exclude=exclude, targets=targets)
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.label = label

    def __call__(
        self, clip_evaluations: Sequence[ClipEval]
    ) -> Dict[str, float]:
        y_true, y_score, num_positives = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        class_scores = {
            class_name: average_precision(
                y_true[class_name],
                y_score[class_name],
                num_positives=num_positives[class_name],
            )
            for class_name in self.targets.class_names
        }

        mean_score = float(
            np.mean([v for v in class_scores.values() if not np.isnan(v)])
        )

        result = {
            f"mean_{self.label}": mean_score,
            **{
                f"{self.label}/{class_name}": score
                for class_name, score in class_scores.items()
                if self.include_class(class_name)
            },
        }
        
        # Compute genus-level metrics if genus information is available
        if self.targets.genus_names is not None and self.targets.class_to_genus is not None:
            genus_y_true = defaultdict(list)
            genus_y_score = defaultdict(list)
            
            # Debug counters
            total_gt_annotations = 0
            total_matches_processed = 0
            cross_genus_confusions = 0
            
            # For each clip, we need to find the best prediction per ground truth
            for clip_eval in clip_evaluations:
                # Group all matches by ground truth ID to find best prediction per GT
                gt_to_best_match = {}
                
                for class_name, matches in clip_eval.matches.items():
                    pred_genus_idx = self.targets.class_to_genus.get(class_name)
                    if pred_genus_idx is None:
                        continue
                    pred_genus = self.targets.genus_names[pred_genus_idx]
                    
                    for m in matches:
                        if m.is_generic and self.ignore_generic:
                            continue
                        
                        # We want predictions that have a true class (i.e., matched to GT)
                        if not m.is_prediction:
                            continue
                        
                        if m.true_class is None:
                            continue
                        
                        total_matches_processed += 1
                        
                        # Use true_class as unique identifier for ground truth
                        # Since we're iterating over predictions, multiple predictions 
                        # might match the same GT, so we use a tuple of (true_class, score) 
                        # Actually, we need a better way to group by GT
                        # For now, use the prediction itself as we want best pred per GT
                        # Let's use gt if available, otherwise use a composite key
                        if m.gt is not None:
                            gt_id = id(m.gt)
                        else:
                            # If no gt object, create key from clip + true_class + rough location
                            gt_id = (id(clip_eval.clip), m.true_class, m.score)
                        
                        # Keep track of the best (highest scoring) prediction for this GT
                        if gt_id not in gt_to_best_match or m.score > gt_to_best_match[gt_id]['score']:
                            gt_to_best_match[gt_id] = {
                                'score': m.score,
                                'pred_genus': pred_genus,
                                'pred_class': class_name,
                                'true_class': m.true_class,
                            }
                
                # Now compute genus accuracy based on best predictions
                for gt_id, best_match in gt_to_best_match.items():
                    total_gt_annotations += 1
                    true_class = best_match['true_class']
                    if true_class and true_class in self.targets.class_to_genus:
                        true_genus_idx = self.targets.class_to_genus[true_class]
                        true_genus = self.targets.genus_names[true_genus_idx]
                        
                        pred_genus = best_match['pred_genus']
                        score = best_match['score']
                        
                        # Check if predicted genus matches true genus
                        is_correct_genus = (true_genus == pred_genus)
                        if not is_correct_genus:
                            cross_genus_confusions += 1
                            # Debug: print first few confusions
                            if cross_genus_confusions <= 3:
                                print(f"Cross-genus confusion: predicted {best_match['pred_class']} (genus {pred_genus}) "
                                      f"but true is {true_class} (genus {true_genus}), score={score:.3f}")
                        
                        genus_y_true[true_genus].append(is_correct_genus)
                        genus_y_score[true_genus].append(score)
            
            # Print debug summary
            print(f"Genus eval debug: {total_gt_annotations} GTs, {total_matches_processed} matches, "
                  f"{cross_genus_confusions} cross-genus confusions ({100*cross_genus_confusions/max(1,total_gt_annotations):.1f}%)")
            
            # Compute average precision per genus
            genus_scores = {}
            for genus_name in self.targets.genus_names:
                if genus_name in genus_y_true and len(genus_y_true[genus_name]) > 0:
                    num_positives = sum(genus_y_true[genus_name])
                    if num_positives > 0:
                        genus_scores[genus_name] = average_precision(
                            genus_y_true[genus_name],
                            genus_y_score[genus_name],
                            num_positives=num_positives,
                        )
            
            genus_mean_score = float(
                np.mean([v for v in genus_scores.values() if not np.isnan(v)])
            ) if genus_scores else np.nan
            
            result[f"genus/mean_{self.label}"] = genus_mean_score
            for genus_name, score in genus_scores.items():
                result[f"genus/{self.label}/{genus_name}"] = score
        
        return result

    @classification_metrics.register(ClassificationAveragePrecisionConfig)
    @staticmethod
    def from_config(
        config: ClassificationAveragePrecisionConfig,
        targets: TargetProtocol,
    ):
        return ClassificationAveragePrecision(
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            label=config.label,
            include=config.include,
            exclude=config.exclude,
        )


class ClassificationROCAUCConfig(BaseClassificationConfig):
    name: Literal["roc_auc"] = "roc_auc"
    label: str = "roc_auc"
    ignore_non_predictions: bool = True
    ignore_generic: bool = True


class ClassificationROCAUC(BaseClassificationMetric):
    def __init__(
        self,
        targets: TargetProtocol,
        ignore_non_predictions: bool = True,
        ignore_generic: bool = True,
        label: str = "roc_auc",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.targets = targets
        self.ignore_non_predictions = ignore_non_predictions
        self.ignore_generic = ignore_generic
        self.label = label
        self.include = include
        self.exclude = exclude

    def __call__(
        self, clip_evaluations: Sequence[ClipEval]
    ) -> Dict[str, float]:
        y_true, y_score, _ = _extract_per_class_metric_data(
            clip_evaluations,
            ignore_non_predictions=self.ignore_non_predictions,
            ignore_generic=self.ignore_generic,
        )

        class_scores = {
            class_name: float(
                metrics.roc_auc_score(
                    y_true[class_name],
                    y_score[class_name],
                )
            )
            for class_name in self.targets.class_names
        }

        mean_score = float(
            np.mean([v for v in class_scores.values() if v != np.nan])
        )

        return {
            f"mean_{self.label}": mean_score,
            **{
                f"{self.label}/{class_name}": score
                for class_name, score in class_scores.items()
                if self.include_class(class_name)
            },
        }

    @classification_metrics.register(ClassificationROCAUCConfig)
    @staticmethod
    def from_config(
        config: ClassificationROCAUCConfig, targets: TargetProtocol
    ):
        return ClassificationROCAUC(
            targets=targets,
            ignore_non_predictions=config.ignore_non_predictions,
            ignore_generic=config.ignore_generic,
            label=config.label,
        )


ClassificationMetricConfig = Annotated[
    Union[
        ClassificationAveragePrecisionConfig,
        ClassificationROCAUCConfig,
    ],
    Field(discriminator="name"),
]


def build_classification_metric(
    config: ClassificationMetricConfig,
    targets: TargetProtocol,
) -> ClassificationMetric:
    return classification_metrics.build(config, targets)


def _extract_per_class_metric_data(
    clip_evaluations: Sequence[ClipEval],
    ignore_non_predictions: bool = True,
    ignore_generic: bool = True,
):
    y_true = defaultdict(list)
    y_score = defaultdict(list)
    num_positives = defaultdict(lambda: 0)

    for clip_eval in clip_evaluations:
        for class_name, matches in clip_eval.matches.items():
            for m in matches:
                # Exclude matches with ground truth sounds where the class
                # is unknown
                if m.is_generic and ignore_generic:
                    continue

                is_class = m.true_class == class_name

                if is_class:
                    num_positives[class_name] += 1

                # Ignore matches that don't correspond to a prediction
                if not m.is_prediction and ignore_non_predictions:
                    continue

                y_true[class_name].append(is_class)
                y_score[class_name].append(m.score)

    return y_true, y_score, num_positives


def compute_precision_recall_curves(
    clip_evaluations: Sequence[ClipEval],
    ignore_non_predictions: bool = True,
    ignore_generic: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    y_true, y_score, num_positives = _extract_per_class_metric_data(
        clip_evaluations,
        ignore_non_predictions=ignore_non_predictions,
        ignore_generic=ignore_generic,
    )

    return {
        class_name: compute_precision_recall(
            y_true[class_name],
            y_score[class_name],
            num_positives=num_positives[class_name],
        )
        for class_name in y_true
    }
