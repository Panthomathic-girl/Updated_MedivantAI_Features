# app/order_pattern_forecasting/customer_forecasting/schema.py

from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime

class OrderRecord(BaseModel):
    order_date: datetime | str  # Accepts string or datetime
    customer_id: str
    product_id: str
    quantity: int

class CustomerOrder(BaseModel):
    customer_id: str
    products: Dict[str, int]


class CustomerForecastResponse(BaseModel):
    forecast_month: str
    total_predicted_orders: int
    total_customers_expected_to_order: int
    customer_orders: List[CustomerOrder]
    generated_on: str
    message: str


class PredictRequest(BaseModel):
    year: int
    month: int


class TrainRequest(BaseModel):
    orders: List[OrderRecord]
    