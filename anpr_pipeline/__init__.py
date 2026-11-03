import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from anpr_pipeline.base import (
    BaseVehicleTracker,
    BaseMetadataEstimator,
    BasePlateDetector,
    BaseOCREngine,
    BaseGrammarValidator,
    BaseTemporalFusion
)
from anpr_pipeline.pipeline import ANPRPipeline
from anpr_pipeline.grammar_validator import IndianRTOValidator
from anpr_pipeline.temporal_fusion import TemporalFusion
from anpr_pipeline.ocr_engine import EasyOCREngine, GotOCREngine, get_ocr_engine

__all__ = [
    "ANPRPipeline",
    "IndianRTOValidator",
    "TemporalFusion",
    "EasyOCREngine",
    "GotOCREngine",
    "get_ocr_engine"
]
