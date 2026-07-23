from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent / "telecom_churn_stack.pkl"
model = None

# Exact feature order used by the notebook:
FEATURE_ORDER = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# LabelEncoder mappings reproduced from the notebook output.
BINARY_MAP = {"No": 0, "Yes": 1}

INTERNET_OPTION_MAP = {
    "No": 0,
    "No internet service": 1,
    "Yes": 2,
}

CONTRACT_MAP = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}

PAYMENT_METHOD_MAP = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

# Min and max values shown in the notebook before MinMaxScaler.
SCALING = {
    "tenure": (0.0, 72.0),
    "MonthlyCharges": (18.25, 118.75),
    "TotalCharges": (0.0, 8684.80),
}


class CustomerInput(BaseModel):
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: float = Field(ge=0, le=72)
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


def scale(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def require_mapping(value: str, mapping: dict[str, int], field: str) -> int:
    if value not in mapping:
        allowed = ", ".join(mapping.keys())
        raise ValueError(f"{field} must be one of: {allowed}")
    return mapping[value]


def preprocess(customer: CustomerInput) -> np.ndarray:
    values = {
        "SeniorCitizen": customer.SeniorCitizen,
        "Partner": require_mapping(customer.Partner, BINARY_MAP, "Partner"),
        "Dependents": require_mapping(customer.Dependents, BINARY_MAP, "Dependents"),
        "tenure": scale(customer.tenure, *SCALING["tenure"]),
        "OnlineSecurity": require_mapping(
            customer.OnlineSecurity, INTERNET_OPTION_MAP, "OnlineSecurity"
        ),
        "OnlineBackup": require_mapping(
            customer.OnlineBackup, INTERNET_OPTION_MAP, "OnlineBackup"
        ),
        "DeviceProtection": require_mapping(
            customer.DeviceProtection, INTERNET_OPTION_MAP, "DeviceProtection"
        ),
        "TechSupport": require_mapping(
            customer.TechSupport, INTERNET_OPTION_MAP, "TechSupport"
        ),
        "Contract": require_mapping(customer.Contract, CONTRACT_MAP, "Contract"),
        "PaperlessBilling": require_mapping(
            customer.PaperlessBilling, BINARY_MAP, "PaperlessBilling"
        ),
        "PaymentMethod": require_mapping(
            customer.PaymentMethod, PAYMENT_METHOD_MAP, "PaymentMethod"
        ),
        "MonthlyCharges": scale(
            customer.MonthlyCharges, *SCALING["MonthlyCharges"]
        ),
        "TotalCharges": scale(customer.TotalCharges, *SCALING["TotalCharges"]),
    }

    return np.array([[values[name] for name in FEATURE_ORDER]], dtype=float)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "telecom_churn_stack.pkl is missing. "
            "Run the export cell in the notebook first."
        )

    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="Telecom Customer Churn API",
    description="API for the stacking classifier trained in Telecom_churn(1).ipynb.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Telecom churn prediction API is running.",
        "docs": "/docs",
        "model": "StackingClassifier",
        "features": FEATURE_ORDER,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        features = preprocess(customer)
        prediction = int(model.predict(features)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])

        return {
            "prediction": prediction,
            "churn": bool(prediction),
            "label": (
                "Customer is likely to churn"
                if prediction == 1
                else "Customer is unlikely to churn"
            ),
            "churn_probability": (
                round(probability, 4) if probability is not None else None
            ),
            "churn_percentage": (
                round(probability * 100, 2) if probability is not None else None
            ),
        }

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
