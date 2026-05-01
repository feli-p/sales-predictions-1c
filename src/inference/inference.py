"""
inference.py

Carga el modelo entrenado y genera predicciones para el archivo de envío
submission.csv.
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference: generate submission.csv")
    parser.add_argument("--model-path", type=str, default="artifacts/model.joblib")
    parser.add_argument("--inference-dir", type=str, default="data/inference")
    parser.add_argument("--output-path", type=str, default="data/predictions/submission.csv")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    logger = setup_logger("prediccion")
    start_time = time.time()

    logger.info("Iniciando predicción...")

    # src/inference/inference.py -> parents[2] = raíz del repo
    project_root = Path(__file__).resolve().parents[2]

    args = parse_args()

    model_path = project_root / args.model_path
    inference_dir = project_root / args.inference_dir
    out_path = project_root / args.output_path

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        logger.error("Modelo no encontrado: %s", model_path)
        raise

    x_test_path = inference_dir / "xtest.pkl"
    test_path = inference_dir / "test.pkl"

    if not x_test_path.exists():
        logger.error("No se encontró: %s", x_test_path)
        raise FileNotFoundError(x_test_path)

    if not test_path.exists():
        logger.error("No se encontró: %s", test_path)
        raise FileNotFoundError(test_path)

    x_test = pd.read_pickle(x_test_path)
    test = pd.read_pickle(test_path)

    logger.info("Cargado %s (rows=%d, cols=%d)", x_test_path.name, len(x_test), x_test.shape[1])
    logger.info("Cargado %s (rows=%d, cols=%d)", test_path.name, len(test), test.shape[1])

    if "ID" not in test.columns:
        logger.error("La columna 'ID' no existe en %s", test_path.name)
        raise ValueError("Missing required column: ID")

    pred_test = model.predict(x_test)
    pred_test = np.clip(pred_test, args.clip_min, args.clip_max)

    submission = pd.DataFrame({"ID": test["ID"], "item_cnt_month": pred_test})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    logger.info("Predicciones guardadas: %s (rows=%d)", out_path.name, len(submission))
    logger.info("Fin predicción. Tiempo: %.2fs", time.time() - start_time)


if __name__ == "__main__":
    main()