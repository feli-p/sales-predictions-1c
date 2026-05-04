import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    prep_dir = project_root / "data/prep"
    
    base = pd.read_pickle(prep_dir / "base.pkl")
    monthly = pd.read_pickle(prep_dir / "monthly.pkl")
    
    lag1 = monthly[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy()
    lag1["date_block_num"] += 1
    lag1 = lag1.rename(columns={"item_cnt_month": "naive_pred"})
    
    df_baseline = base.merge(lag1, on=["date_block_num", "shop_id", "item_id"], how="left")
    df_baseline["naive_pred"] = df_baseline["naive_pred"].fillna(0).clip(0, 20)
    
    val_mask = df_baseline["date_block_num"] == 33
    val_data = df_baseline[val_mask].dropna(subset=["item_cnt_month"])
    
    rmse_naive = np.sqrt(mean_squared_error(val_data["item_cnt_month"], val_data["naive_pred"]))
    
    print(f"--- Baseline RMSE ---")
    print(f"Naive RMSE: {rmse_naive:.6f}")

    metrics_path = Path(__file__).resolve().parent / "metrica-baseline.json"
    metrics = {
        "model_name": "Modelo Naive (Lag-1)",
        "rmse": rmse_naive,
        "description": "Predicción que copia las ventass del mes anterior."
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    test_mask = df_baseline["date_block_num"] == 34
    test_preds = df_baseline[test_mask].copy()
    
    test_raw = pd.read_csv(project_root / "data/raw/test.csv")
    submission = test_raw.merge(test_preds[["shop_id", "item_id", "naive_pred"]], 
                                on=["shop_id", "item_id"], how="left")
    
    submission = submission.rename(columns={"naive_pred": "item_cnt_month"})[["ID", "item_cnt_month"]]
    submission["item_cnt_month"] = submission["item_cnt_month"].fillna(0)
    
    out_dir = project_root / "data/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / "baseline_naive.csv", index=False)
    print(f"Archivo guardado en: {out_dir / 'baseline_naive.csv'}")

if __name__ == "__main__":
    main()