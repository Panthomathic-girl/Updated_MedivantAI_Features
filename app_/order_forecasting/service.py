# app/order_pattern_forecasting/customer_forecasting/service.py

from fastapi import HTTPException
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from lightgbm import LGBMClassifier, LGBMRegressor
from config import OrderConfig


class ModelUtils:
    _model_cache: Dict[str, Any] | None = None

    @staticmethod
    def load_and_clean_from_df(df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame received from JSON input."""
        df = df.copy()

        df[OrderConfig.COL_ORDER_DATE] = pd.to_datetime(df[OrderConfig.COL_ORDER_DATE], errors='coerce')
        df = df.dropna(subset=[
            OrderConfig.COL_ORDER_DATE,
            OrderConfig.COL_CUSTOMER_ID,
            OrderConfig.COL_PRODUCT_ID,
            OrderConfig.COL_QUANTITY
        ])
        df = df[df[OrderConfig.COL_QUANTITY] > 0].copy()
        df = df.sort_values(OrderConfig.COL_ORDER_DATE).reset_index(drop=True)

        if df.empty:
            print("Warning: No clean order records remained after filtering.")
        else:
            print(f"Cleaned data: {len(df):,} valid orders from "
                  f"{df[OrderConfig.COL_ORDER_DATE].min().date()} "
                  f"to {df[OrderConfig.COL_ORDER_DATE].max().date()}")

        return df

    @classmethod
    def _load_model(cls) -> Dict[str, Any]:
        if cls._model_cache is not None:
            return cls._model_cache

        model_file = Path(OrderConfig.ORDER_FORECAST_MODEL_FILE)
        if not model_file.exists():
            raise HTTPException(
                status_code=424,
                detail=(
                    "Trained model not found. Please train the model first by sending order data to:\n"
                    "→ POST /order/train"
                )
            )

        print(f"[{datetime.now():%H:%M:%S}] Loading model from {model_file}...")
        try:
            cls._model_cache = joblib.load(model_file)
            pkg = cls._model_cache
            print(f"Model loaded! Version: {pkg.get('model_version', 'unknown')}")
            return pkg
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model (corrupted or incompatible): {str(e)}"
            ) from e


class Training:
    @staticmethod
    def _generate_training_samples(df: pd.DataFrame, cust_to_num: Dict[str, int], prod_to_num: Dict[str, int]) -> tuple[list, list, list, dict]:
        if df.empty:
            raise ValueError("Empty dataframe for training samples.")

        df = df.copy()
        df['order_date'] = df[OrderConfig.COL_ORDER_DATE]
        df['yrmnth'] = df['order_date'].dt.to_period('M')
        all_months = sorted(df['yrmnth'].unique())

        if len(all_months) == 0:
            raise ValueError("No valid months in data.")

        print(f"Data spans {len(all_months)} months: {all_months[0]} to {all_months[-1]}")

        X_train = []
        y_order = []
        y_qty = []
        metadata = {
            'last_order_dates': {},
            'frequency': {},
            'avg_qty_last_3m': {},
            'tenure_days': {}
        }

        for idx, target_period in enumerate(all_months[:-1]):
            target_date = target_period.to_timestamp()
            next_month_date = target_date + pd.offsets.MonthBegin(1)

            historical = df[df['order_date'] < target_date]
            target_month_df = df[(df['order_date'] >= target_date) & (df['order_date'] < next_month_date)]

            if historical.empty:
                continue

            actual_pairs = set(zip(target_month_df['customer_id'], target_month_df['product_id']))
            qty_map = dict(zip(zip(target_month_df['customer_id'], target_month_df['product_id']),
                               target_month_df['quantity']))

            cp_stats = historical.groupby(['customer_id', 'product_id']).agg(
                last_order=('order_date', 'max'),
                first_order=('order_date', 'min'),
                total_orders=('order_date', 'count'),
                total_qty=('quantity', 'sum'),
                qty_last_3m=('quantity', lambda x: x[historical['order_date'] >= target_date - pd.DateOffset(months=3)].sum()),
                n_last_3m=('order_date', lambda x: (x >= target_date - pd.DateOffset(months=3)).sum())
            ).reset_index()

            for _, row in cp_stats.iterrows():
                cust_id = row['customer_id']
                prod_id = row['product_id']
                if cust_id not in cust_to_num or prod_id not in prod_to_num:
                    continue

                key = (cust_id, prod_id)
                recency_days = (target_date - row['last_order']).days
                if recency_days > 1095:
                    continue

                tenure_days = max(1, (row['last_order'] - row['first_order']).days + 1)
                frequency = row['total_orders'] / (tenure_days / 30.0)
                avg_qty_last_3m = (
                    row['qty_last_3m'] / max(1, row['n_last_3m'])
                    if row['n_last_3m'] > 0 else
                    row['total_qty'] / max(1, row['total_orders'])
                )

                metadata['last_order_dates'][key] = row['last_order'].isoformat()
                metadata['frequency'][key] = frequency
                metadata['avg_qty_last_3m'][key] = avg_qty_last_3m
                metadata['tenure_days'][key] = tenure_days

                X_train.append([
                    cust_to_num[cust_id],
                    prod_to_num[prod_id],
                    recency_days,
                    np.exp(-recency_days / 90.0),
                    frequency,
                    avg_qty_last_3m,
                    int(recency_days <= 120),
                    tenure_days
                ])

                ordered = 1 if key in actual_pairs else 0
                y_order.append(ordered)
                if ordered:
                    y_qty.append(max(1, int(qty_map.get(key, 1))))

        return X_train, y_order, y_qty, metadata

    @staticmethod
    def train_model(df: pd.DataFrame) -> Dict[str, Any]:
        df['yrmnth'] = df[OrderConfig.COL_ORDER_DATE].dt.to_period('M')
        all_months = sorted(df['yrmnth'].unique())

        if len(all_months) < 2:
            raise ValueError("Need at least 2 months of data for full training.")

        print(f"Training from scratch on {len(all_months)} months: {all_months[0]} to {all_months[-1]}")

        cust_to_num = {c: i for i, c in enumerate(sorted(df['customer_id'].unique()))}
        prod_to_num = {p: i for i, p in enumerate(sorted(df['product_id'].unique()))}
        num_to_cust = {i: c for c, i in cust_to_num.items()}
        num_to_prod = {i: p for p, i in prod_to_num.items()}

        X_train, y_order, y_qty, metadata = Training._generate_training_samples(df, cust_to_num, prod_to_num)

        clf_params = {
            'n_estimators': 700,
            'learning_rate': 0.05,
            'max_depth': 10,
            'min_child_samples': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        clf = LGBMClassifier(**clf_params)
        clf.fit(X_train, y_order)

        X_pos = [x for x, o in zip(X_train, y_order) if o == 1]
        if not X_pos:
            raise ValueError("No positive samples for regressor training.")

        reg_params = {
            'n_estimators': 700,
            'learning_rate': 0.05,
            'max_depth': 10,
            'min_child_samples': 5,
            'random_state': 42,
            'verbose': -1
        }
        reg = LGBMRegressor(**reg_params)
        reg.fit(X_pos, y_qty)

        feature_names = [
            'cust_num', 'prod_num', 'recency_days', 'recency_score',
            'frequency', 'avg_qty_last_3m', 'is_active', 'tenure_days'
        ]

        return {
            'classifier': clf,
            'regressor': reg,
            'cust_to_num': cust_to_num,
            'prod_to_num': prod_to_num,
            'cust_mapping': num_to_cust,
            'prod_mapping': num_to_prod,
            'last_order_dates': metadata['last_order_dates'],
            'frequency': metadata['frequency'],
            'avg_qty_last_3m': metadata['avg_qty_last_3m'],
            'tenure_days': metadata['tenure_days'],
            'feature_names': feature_names,
            'trained_on': datetime.now().isoformat(),
            'data_period': f"{all_months[0]} → {all_months[-1]}",
            'total_samples': len(X_train),
            'positive_samples': len(y_qty),
            'model_version': '1.0.0',
        }

    @staticmethod
    def fine_tune_with_new_data(new_df: pd.DataFrame) -> dict:
        model_path = Path(OrderConfig.ORDER_FORECAST_MODEL_FILE)
        existing_package = joblib.load(model_path)

        print(f"Fine-tuning existing model with {len(new_df):,} new records...")

        old_cust = set(existing_package['cust_to_num'].keys())
        old_prod = set(existing_package['prod_to_num'].keys())
        new_cust = set(new_df['customer_id'].unique())
        new_prod = set(new_df['product_id'].unique())
        all_cust = sorted(old_cust | new_cust)
        all_prod = sorted(old_prod | new_prod)

        cust_to_num = {c: i for i, c in enumerate(all_cust)}
        prod_to_num = {p: i for i, p in enumerate(all_prod)}
        num_to_cust = {i: c for c, i in cust_to_num.items()}
        num_to_prod = {i: p for p, i in prod_to_num.items()}

        X_train, y_order, y_qty, new_metadata = Training._generate_training_samples(new_df, cust_to_num, prod_to_num)

        clf_params = {
            'n_estimators': 700,
            'learning_rate': 0.05,
            'max_depth': 10,
            'min_child_samples': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        clf = LGBMClassifier(**clf_params)
        clf.fit(X_train, y_order, init_model=existing_package['classifier'])

        X_pos = [x for x, o in zip(X_train, y_order) if o == 1]
        reg_params = {
            'n_estimators': 700,
            'learning_rate': 0.05,
            'max_depth': 10,
            'min_child_samples': 5,
            'random_state': 42,
            'verbose': -1
        }
        reg = LGBMRegressor(**reg_params)
        if X_pos:
            reg.fit(X_pos, y_qty, init_model=existing_package['regressor'])
        else:
            reg = existing_package['regressor']
            print("No new positive samples, keeping existing regressor.")

        last_order_dates = {**existing_package['last_order_dates'], **new_metadata['last_order_dates']}
        frequency = {**existing_package['frequency'], **new_metadata['frequency']}
        avg_qty_last_3m = {**existing_package['avg_qty_last_3m'], **new_metadata['avg_qty_last_3m']}
        tenure_days = {**existing_package['tenure_days'], **new_metadata['tenure_days']}

        model_package = {
            'classifier': clf,
            'regressor': reg,
            'cust_to_num': cust_to_num,
            'prod_to_num': prod_to_num,
            'cust_mapping': num_to_cust,
            'prod_mapping': num_to_prod,
            'last_order_dates': last_order_dates,
            'frequency': frequency,
            'avg_qty_last_3m': avg_qty_last_3m,
            'tenure_days': tenure_days,
            'feature_names': existing_package['feature_names'],
            'trained_on': datetime.now().isoformat(),
            'data_period': existing_package['data_period'],
            'total_samples': existing_package['total_samples'] + len(X_train),
            'positive_samples': existing_package['positive_samples'] + len(y_qty),
            'model_version': '1.0.0',
        }

        joblib.dump(model_package, model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        ModelUtils._model_cache = None

        print(f"Fine-tune complete → {size_mb:.2f} MB")

        return {
            "new_model_path": str(model_path),
            "total_samples": model_package['total_samples'],
            "positive_samples": model_package['positive_samples'],
            "model_size_mb": size_mb,
            "new_records_added": len(new_df),
        }


class Prediction:
    @staticmethod
    def predict_raw(year: int, month: int) -> dict:
        try:
            pkg = ModelUtils._load_model()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected model load error: {str(e)}")

        clf = pkg['classifier']
        reg = pkg['regressor']
        cust_to_num = pkg['cust_to_num']
        prod_to_num = pkg['prod_to_num']

        target_date = pd.Timestamp(f"{year}-{month:02d}-01")
        predictions = []
        total = 0

        all_historical_qty = [
            pkg['avg_qty_last_3m'].get(k, 1) * pkg['frequency'].get(k, 0.05) * 30
            for k in pkg['last_order_dates']
        ]
        qty_cap = int(np.percentile(all_historical_qty, 97)) if all_historical_qty else 120
        qty_cap = max(60, qty_cap)

        for cust_id in cust_to_num:
            for prod_id in prod_to_num:
                key = (cust_id, prod_id)
                last_str = pkg['last_order_dates'].get(key)
                if not last_str:
                    continue

                last_date = pd.Timestamp(last_str)
                recency_days = (target_date - last_date).days
                if recency_days > 5475:
                    continue

                recency_score = np.exp(-recency_days / 365.0)
                frequency = pkg['frequency'].get(key, 0.01)
                avg_qty = pkg['avg_qty_last_3m'].get(key, 1.0)
                is_active = int(recency_days <= 365)
                tenure = pkg['tenure_days'].get(key, 30)

                X = np.array([[
                    cust_to_num[cust_id], prod_to_num[prod_id], recency_days,
                    recency_score, frequency, avg_qty, is_active, tenure
                ]])

                prob = clf.predict_proba(X)[0, 1]
                threshold = 0.40 if recency_days > 2000 else 0.50
                if prob < threshold:
                    continue

                raw_qty = reg.predict(X)[0]
                qty = max(1, int(round(raw_qty * (1 + recency_days / 10000))))
                qty = min(qty, qty_cap)

                total += qty
                predictions.append({
                    "customer_id": cust_id,
                    "product_id": prod_id,
                    "predicted_quantity": qty,
                    "probability": round(prob, 4)
                })

        predictions.sort(key=lambda x: x["probability"], reverse=True)
        print(f"Prediction complete: {len(predictions)} items for {year}-{month:02d}")

        return {
            "predictions": predictions,
            "total_predicted_orders": int(total),
            "active_pairs": len(predictions),
            "quantity_cap_used": qty_cap
        }