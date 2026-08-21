from pipeline.ocr_model.ppocr import (MODEL_CHOICES, MODEL_NAME, OCRReader,
                                      PPOCRReader)
from pipeline.ocr_model.vintern import VinternReader

__all__ = ["OCRReader", "PPOCRReader", "VinternReader", "MODEL_NAME",
           "MODEL_CHOICES"]
