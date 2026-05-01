"""
train.py

Entrena un modelo Ridge para predecir ventas mensuales.
Carga datos preparados (monthly.pkl y base.pkl), crea features (lags, month, avg_price),
evalúa RMSE en el último bloque y guarda el modelo en artifacts/model.joblib.
"""

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.common.logging_utils import setup_logger

CLIP_MIN = 0
CLIP_MAX = 20

FEATURE_COLUMNS = [
    "shop_id",
    "item_id",
    "item_category_id",
    "month",
    "lag1_cnt",
    "lag12_cnt",
    "avg_price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SageMaker training job")

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )

    parser.add_argument("--monthly-file", type=str, default="monthly.pkl")
    parser.add_argument("--base-file", type=str, default="base.pkl")
    parser.add_argument("--alpha", type=float, default=1.0)

    return parser.parse_args()


def crear_lags(base, monthly) -> pd.DataFrame:
    lag1 = monthly[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy()
    lag1["date_block_num"] += 1
    lag1 = lag1.rename(columns={"item_cnt_month": "lag1_cnt"})
    base = base.merge(lag1, on=["date_block_num", "shop_id", "item_id"], how="left")
    base["lag1_cnt"] = base["lag1_cnt"].fillna(0)

    lag12 = monthly[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy()
    lag12["date_block_num"] += 12
    lag12 = lag12.rename(columns={"item_cnt_month": "lag12_cnt"})
    base = base.merge(lag12, on=["date_block_num", "shop_id", "item_id"], how="left")
    base["lag12_cnt"] = base["lag12_cnt"].fillna(0)

    base["month"] = base["date_block_num"] % 12
    return base


def impute_avg_price(base: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    item_avg_price = monthly.groupby("item_id")["avg_price"].mean()
    base["avg_price"] = base["avg_price"].fillna(base["item_id"].map(item_avg_price))
    base["avg_price"] = base["avg_price"].fillna(monthly["avg_price"].median())
    return base


def main():
    logger = setup_logger("train")
    args = parse_args()
    start = time.time()

    train_dir = Path(args.train)
    model_dir = Path(args.model_dir)
    output_data_dir = Path(args.output_data_dir)

    monthly_path = train_dir / args.monthly_file
    base_path = train_dir / args.base_file

    if not monthly_path.exists():
        raise FileNotFoundError(f"No se encontró {monthly_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"No se encontró {base_path}")

    monthly = pd.read_pickle(monthly_path)
    base = pd.read_pickle(base_path)

    last_block = int(monthly["date_block_num"].max())

    base = crear_lags(base, monthly)
    base = impute_avg_price(base, monthly)

    train_data = (
        base[base["date_block_num"] <= last_block]
        .dropna(subset=["item_cnt_month"])
        .copy()
    )

    features = train_data[FEATURE_COLUMNS].astype(float)
    target = train_data["item_cnt_month"].astype(float)

    is_train = train_data["date_block_num"] < last_block
    is_valid = train_data["date_block_num"] == last_block

    if args.alpha == 1.0:
        ts_split = TimeSeriesSplit(n_splits=3)
        param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]}

        grid = GridSearchCV(
            estimator=Ridge(random_state=0),
            param_grid=param_grid,
            cv=ts_split,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(features[is_train], target[is_train])
        model = grid.best_estimator_
        best_alpha = float(grid.best_params_["alpha"])
    else:
        best_alpha = float(args.alpha)
        model = Ridge(alpha=best_alpha, random_state=0)
        model.fit(features[is_train], target[is_train])

    pred_valid = model.predict(features[is_valid])
    pred_valid = np.clip(pred_valid, CLIP_MIN, CLIP_MAX)

    rmse = float(np.sqrt(mean_squared_error(target[is_valid], pred_valid)))

    model_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "clip_min": CLIP_MIN,
            "clip_max": CLIP_MAX,
            "alpha": best_alpha,
            "rmse_valid": rmse,
        },
        model_dir / "metadata.joblib",
    )

    logger.info("Modelo guardado en %s", model_dir)
    logger.info("RMSE valid=%.6f", rmse)
    logger.info("Tiempo total %.2fs", time.time() - start)


if __name__ == "__main__":
    main()