import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.predict import AnxietyPredictor


app = FastAPI(title="AI-Based Exam Anxiety Detector API", version="1.0.0")
predictor = None


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)


def setup_logger() -> logging.Logger:
    log_dir = ROOT_DIR / "backend"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "predictions.log"

    logger = logging.getLogger("prediction_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


@app.on_event("startup")
def startup_event():
    global predictor
    predictor = AnxietyPredictor(str(ROOT_DIR))


@app.get("/")
def root():
    return {"message": "AI Exam Anxiety Detector API is running."}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        result = predictor.predict(request.text)
        logger.info(
            "text=%s | predicted=%s | confidence=%.4f",
            request.text[:180].replace("\n", " "),
            result["anxiety_level"],
            result["confidence"],
        )
        return result
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {ex}") from ex
