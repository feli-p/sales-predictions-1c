import argparse
import os
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

from src.common.logging_utils import setup_logger

# To-do: Agregarlos como argumentos
AWS_REGION = "us-east-1"
BUCKET_NAME = "sales-predictions-1c"
S3_PREFIX = "data/silver"
GLUE_DB_NAME = "sales-1c-db"

# To-Do: Agregar data-typespara las tablas.

FILES = ["sales_train.csv", "items_en.csv", "item_categories_en.csv", "shops_en.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETL silver layer: Agrupa data por mes y almacena parquets en AWS."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directorio de entrada con CSVs raw (relativo a la raíz del repo).",
    )
    return parser.parse_args()


def main():
    logger = setup_logger("etl silver")
    logger.info("Iniciando capa Silver...")
    args = parse_args()

    session = boto3.Session(region_name=AWS_REGION)

    # En un futuro los archivos deberán obtenerse en la bronze layer en S3
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.data_dir

    # Cargar archivos raw
    tablas: dict[str, pd.DataFrame] = {}
    for filename in FILES:
        path = data_dir / filename
        if not path.exists():
            logger.error("No se encontró: %s", path)
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        tablas[filename] = df
        logger.info("Cargado %s (rows=%d, cols=%d)", path.name, len(df), df.shape[1])

    df_sales = tablas["sales_train.csv"].copy()
    df_shops = tablas["shops_en.csv"].copy()
    df_items = tablas["items_en.csv"].copy()
    df_categories = tablas["item_categories_en.csv"].copy()

    # Agregar ventas mensualmente
    logger.info("Agrupando ventas diarias en mensuales...")
    df_sales_monthly = df_sales.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(
        item_cnt_month=("item_cnt_day", "sum"),
        item_price_avg=("item_price", "mean"),
        item_price_min=("item_price", "min"),
        item_price_max=("item_price", "max"),
    )
    df_sales_monthly["item_price_avg"] = df_sales_monthly["item_price_avg"].round(2)
    logger.info("Ventas diarias: (rows=%d, cols=%d)", len(df_sales), df_sales.shape[1])
    logger.info(
        "Ventas mensuales: (rows=%d, cols=%d)",
        len(df_sales_monthly),
        df_sales_monthly.shape[1],
    )

    # Cargar tablas a S3 y registrándolas en Glue
    logger.info("Subiendo Parquets a S3 y registrando en Glue")
    wr.catalog.create_database(name=GLUE_DB_NAME, exist_ok=True)

    result = wr.s3.to_parquet(
        df=df_sales_monthly,
        path=f"s3://{BUCKET_NAME}/{S3_PREFIX}/sales_monthly/",
        dataset=True,
        database=GLUE_DB_NAME,
        table="sales_monthly",
        partition_cols=["date_block_num"],
        mode="overwrite",
        boto3_session=session,
    )
    logger.info("\t Archivos escritos: %s", len(result["paths"]))

    result = wr.s3.to_parquet(
        df=df_shops,
        path=f"s3://{BUCKET_NAME}/{S3_PREFIX}/shops/",
        dataset=True,
        database=GLUE_DB_NAME,
        table="shops",
        mode="overwrite",
        boto3_session=session,
    )
    logger.info("\t Archivos escritos: %s", len(result["paths"]))

    result = wr.s3.to_parquet(
        df=df_items,
        path=f"s3://{BUCKET_NAME}/{S3_PREFIX}/items/",
        dataset=True,
        database=GLUE_DB_NAME,
        table="items",
        mode="overwrite",
        boto3_session=session,
    )
    logger.info("\t Archivos escritos: %s", len(result["paths"]))

    result = wr.s3.to_parquet(
        df=df_categories,
        path=f"s3://{BUCKET_NAME}/{S3_PREFIX}/item_categories/",
        dataset=True,
        database=GLUE_DB_NAME,
        table="item_categories",
        mode="overwrite",
        boto3_session=session,
    )
    logger.info("\t Archivos escritos: %s", len(result["paths"]))

    logger.info("ETL finalizado...")


if __name__ == "__main__":
    main()
