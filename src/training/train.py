"""
train.py

Entrena un modelo Ridge para predecir ventas mensuales.
Carga datos preparados (monthly.pkl y base.pkl), crea features (lags, month, avg_price),
evalúa RMSE en el último bloque y guarda el modelo en artifacts/model.joblib.
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Training step: train Ridge model")
    parser.add_argument("--prep-dir", type=str, default="data/prep")

    """ parser.add_argument(
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
    ) """

    parser.add_argument("--monthly-file", type=str, default="monthly.pkl")
    parser.add_argument("--base-file", type=str, default="base.pkl")
    parser.add_argument("--output-path", type=str, default="artifacts/models")
    parser.add_argument("--alpha", type=float, default=1.0)

    return parser.parse_args()


def crear_lags(base, monthly) -> pd.DataFrame:
    """Crea lag features."""
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
    """Rellena los precios faltantes usando promedios de items y medianas mensuales."""
    item_avg_price = monthly.groupby("item_id")["avg_price"].mean()
    base["avg_price"] = base["avg_price"].fillna(base["item_id"].map(item_avg_price))
    base["avg_price"] = base["avg_price"].fillna(monthly["avg_price"].median())
    return base


def main():
    start_time = time.time()
    logger = setup_logger("train")
    logger.info("Iniciando entrenamiento...")

    # src/training/train.py -> parents[2] = raíz del repo
    project_root = Path(__file__).resolve().parents[2]
    args = parse_args()

    prep_dir = project_root / args.prep_dir
    monthly_path = prep_dir / args.monthly_file
    base_path = prep_dir / args.base_file
    model_path = project_root / args.output_path

    if not monthly_path.exists():
        logger.error("No se encontró: %s", monthly_path)
        raise FileNotFoundError(monthly_path)

    if not base_path.exists():
        logger.error("No se encontró: %s", base_path)
        raise FileNotFoundError(base_path)

    monthly = pd.read_pickle(monthly_path)
    base = pd.read_pickle(base_path)

    logger.info(
        "Cargado %s (rows=%d, cols=%d)",
        monthly_path.name,
        len(monthly),
        monthly.shape[1],
    )
    logger.info(
        "Cargado %s (rows=%d, cols=%d)", base_path.name, len(base), base.shape[1]
    )

    last_block = int(monthly["date_block_num"].max())
    logger.info("Último date_block_num: %d", last_block)

    base = crear_lags(base, monthly)
    base = impute_avg_price(base, monthly)

    train_data = (
        base[base["date_block_num"] <= last_block]
        .dropna(subset=["item_cnt_month"])
        .copy()
    )
    logger.info("Train rows (con target): %d", len(train_data))

    features = train_data[FEATURE_COLUMNS].astype(float)
    target = train_data["item_cnt_month"].astype(float)

    is_train = train_data["date_block_num"] < last_block
    is_valid = train_data["date_block_num"] == last_block

    # Se implementa Grid Search sólo si alpha tiene su valor default (alpha = 1.0)
    if args.alpha == 1.0:
        logger.info("Iniciando Grid Search para optimización de hiperparámetro...")
        ts_split = TimeSeriesSplit(n_splits=3)
        param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6]}

        # GridSearch usará RSME para determinar la mejor alpha
        grid_search = GridSearchCV(
            estimator=Ridge(random_state=0),
            param_grid=param_grid,
            cv=ts_split,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(features[is_train], target[is_train])
        model = grid_search.best_estimator_
        best_alpha = float(grid_search.best_params_["alpha"])
        logger.info("Alpha optimizada: %s", best_alpha)
    else:
        logger.info("Entrenando modelo con alpha: %.2f.", args.alpha)
        model = Ridge(alpha=float(args.alpha), random_state=0)
        model.fit(features[is_train], target[is_train])

    pred_valid = model.predict(features[is_valid])
    pred_valid = np.clip(pred_valid, CLIP_MIN, CLIP_MAX)

    rmse = float(np.sqrt(mean_squared_error(target[is_valid], pred_valid)))

    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path / "model.joblib")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "alpha": best_alpha,
        "rmse_valid": rmse,
    }

    with open(model_path / "model-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("RMSE valid=%.6f", rmse)
    logger.info("Modelo guardado en %s", model_path)
    logger.info("Tiempo total %.2fs", time.time() - start_time)


if __name__ == "__main__":
    main()
