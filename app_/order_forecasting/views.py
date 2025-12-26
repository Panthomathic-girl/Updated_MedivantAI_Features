# app/order_pattern_forecasting/customer_forecasting/views.py

from fastapi import APIRouter, HTTPException
import joblib
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from config import OrderConfig
from .schema import CustomerForecastResponse, PredictRequest, TrainRequest, TrainRequest
from .service import Prediction, ModelUtils, Training

router = APIRouter(prefix="/order", tags=["Customer-Level Forecasting"])


@router.post("/train")
async def train_from_json_data(request: TrainRequest):
    if not request.orders:
        raise HTTPException(status_code=400, detail="No order records provided in the request.")

    try:
        print(f"[{datetime.now():%H:%M:%S}] Starting FULL training from {len(request.orders):,} provided order records...")

        # Convert list of OrderRecord models to DataFrame
        data = [order.dict() for order in request.orders]
        df = pd.DataFrame(data)

        # Ensure required columns exist
        required_cols = [
            OrderConfig.COL_ORDER_DATE,
            OrderConfig.COL_CUSTOMER_ID,
            OrderConfig.COL_PRODUCT_ID,
            OrderConfig.COL_QUANTITY
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns in data: {missing}"
            )

        # Clean and prepare data
        df = ModelUtils.load_and_clean_from_df(df)
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid order data after cleaning.")

        # Train model
        model_package = Training.train_model(df)

        # Save model
        model_path = Path(OrderConfig.ORDER_FORECAST_MODEL_FILE)
        joblib.dump(model_package, model_path)

        size_mb = model_path.stat().st_size / (1024 * 1024)

        print(f"SUCCESS! Model trained and saved: {model_path} ({size_mb:.2f} MB)")

        # Clear cache
        ModelUtils._model_cache = None

        return {
            "status": "success",
            "message": "Model trained successfully from provided JSON data!",
            "new_model_path": str(model_path),
            "total_input_records": len(request.orders),
            "valid_records_used": len(df),
            "model_size_mb": round(size_mb, 2),
            "trained_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Training failed: {str(e)}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)
    
    
@router.post("/retrain")
async def retrain_with_new_data(request: TrainRequest):
    if not request.orders:
        raise HTTPException(status_code=400, detail="No new order records provided for retraining.")

    model_path = Path(OrderConfig.ORDER_FORECAST_MODEL_FILE)

    if not model_path.exists():
        raise HTTPException(
            status_code=424,
            detail="No existing model found. Use /order/train first or /order/train_or_update for initial training."
        )

    try:
        print(f"[{datetime.now():%H:%M:%S}] Starting INCREMENTAL retraining with {len(request.orders):,} new records...")

        data = [order.dict() for order in request.orders]
        new_df = pd.DataFrame(data)
        new_df = ModelUtils.load_and_clean_from_df(new_df)
        if new_df.empty:
            raise HTTPException(status_code=400, detail="No valid new order data after cleaning.")

        stats = Training.fine_tune_with_new_data(new_df)

        return {
            "success": True,
            "message": "Model successfully fine-tuned with new data!",
            "new_model_path": stats["new_model_path"],
            "new_records_added": stats["new_records_added"],
            "total_training_samples": stats["total_samples"],
            "positive_samples": stats["positive_samples"],
            "model_size_mb": round(stats["model_size_mb"], 2),
            "retrained_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Retraining failed: {str(e)}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)   


@router.post("/predict", response_model=CustomerForecastResponse)
async def predict_customer_orders(request: PredictRequest):
    year, month = request.year, request.month

    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

    try:
        raw_predictions = Prediction.predict_raw(year, month)
    except HTTPException:
        raise  # Already properly formatted

    CUST_ID = OrderConfig.COL_CUSTOMER_ID
    PROD_ID = OrderConfig.COL_PRODUCT_ID
    PRED_QTY = "predicted_quantity"

    # Aggregate by customer
    customer_dict: Dict[str, Dict[str, int]] = {}
    for pred in raw_predictions["predictions"]:
        cust_id = pred[CUST_ID]
        prod_id = pred[PROD_ID]
        qty = pred[PRED_QTY]
        customer_dict.setdefault(cust_id, {})[prod_id] = qty

    customer_orders = [
        {"customer_id": cust_id, "products": products}
        for cust_id, products in customer_dict.items()
    ]

    return {
        "forecast_month": f"{year}-{month:02d}",
        "total_predicted_orders": int(raw_predictions["total_predicted_orders"]),
        "total_customers_expected_to_order": len(customer_orders),
        "customer_orders": customer_orders[:100],  # Limit response size
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"Forecast generated for {month:02d}/{year}",
    }
    
    
@router.post("/train_or_update")
async def train_or_update_model(request: TrainRequest):
    if not request.orders:
        raise HTTPException(status_code=400, detail="No order records provided in the request.")

    try:
        data = [order.dict() for order in request.orders]
        df = pd.DataFrame(data)

        required_cols = [
            OrderConfig.COL_ORDER_DATE,
            OrderConfig.COL_CUSTOMER_ID,
            OrderConfig.COL_PRODUCT_ID,
            OrderConfig.COL_QUANTITY
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns in data: {missing}")

        df = ModelUtils.load_and_clean_from_df(df)
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid order data after cleaning.")

        model_path = Path(OrderConfig.ORDER_FORECAST_MODEL_FILE)
        input_records = len(request.orders)
        valid_records = len(df)

        if model_path.exists():
            # Fine-tune existing model
            stats = Training.fine_tune_with_new_data(df)
            return {
                "status": "success",
                "mode": "incremental_update",
                "message": "Model successfully updated with new data.",
                "new_model_path": stats["new_model_path"],
                "new_records_added": stats["new_records_added"],
                "total_training_samples": stats["total_samples"],
                "positive_samples": stats["positive_samples"],
                "model_size_mb": round(stats["model_size_mb"], 2),
                "updated_at": datetime.now().isoformat(),
            }
        else:
            # Train from scratch
            model_package = Training.train_model(df)
            joblib.dump(model_package, model_path)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            ModelUtils._model_cache = None

            return {
                "status": "success",
                "mode": "full_training",
                "message": "Model trained from scratch (no previous model found).",
                "new_model_path": str(model_path),
                "total_input_records": input_records,
                "valid_records_used": valid_records,
                "model_size_mb": round(size_mb, 2),
                "trained_at": datetime.now().isoformat(),
            }

    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Train-or-update failed: {str(e)}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)