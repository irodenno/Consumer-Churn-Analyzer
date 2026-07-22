from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


MODEL_PATH = (
    Path(__file__).parent
    / "telecom_churn_pipeline.pkl"
)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "telecom_churn_pipeline.pkl was not found."
        )

    model = joblib.load(MODEL_PATH)

    yield

    model = None


app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "Predicts whether a telecom customer "
        "is likely to churn."
    ),
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class CustomerInput(BaseModel):
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Telecom churn API is running.",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable."
        )

    try:
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        prediction = int(
            model.predict(customer_df)[0]
        )

        probability = float(
            model.predict_proba(customer_df)[0][1]
        )

        return {
            "prediction": prediction,
            "churn": prediction == 1,
            "label": (
                "Likely to churn"
                if prediction == 1
                else "Unlikely to churn"
            ),
            "churn_probability": round(
                probability,
                4
            ),
            "churn_percentage": round(
                probability * 100,
                2
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        ) from error