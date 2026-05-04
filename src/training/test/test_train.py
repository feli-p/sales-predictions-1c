import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.training.train import crear_lags, impute_avg_price, FEATURE_COLUMNS

@pytest.fixture
def datos_prueba():
    """Crea dataframes para usar en los tests."""
    monthly = pd.DataFrame({
        "date_block_num": [0, 0, 11, 11],
        "shop_id": [1, 2, 1, 2],
        "item_id": [10, 10, 10, 10],
        "item_cnt_month": [5, 3, 7, 8],
        "avg_price": [100.0, 110.0, 120.0, np.nan] # Promedio 110
    })
    base = pd.DataFrame({
        "date_block_num": [1, 12],
        "shop_id": [1, 1],
        "item_id": [10, 10],
        "avg_price": [np.nan, np.nan] # Necesitan ser imputados
    })
    return base, monthly


def test_crear_lags(datos_prueba):
    """Revisa que lag1_cnt se mapee de date_block_num 0 a date_block_num 1."""
    base, monthly = datos_prueba
    base = crear_lags(base, monthly)
    
    # Verificar si el bloque 1 tiene el lag del bloque 0
    # Item 10 tiene 5 ventas para la tienda 1.
    bloque_1 = base[base["date_block_num"] == 1]
    assert bloque_1["lag1_cnt"].iloc[0] == 5

    # Verificar si el bloque 12 tiene el lag del bloque 0 y del bloque 11
    bloque_12 = base[base["date_block_num"] == 12]
    assert bloque_12["lag12_cnt"].iloc[0] == 5 # Ventas del bloque 0
    assert bloque_12["lag1_cnt"].iloc[0] == 7  # Ventas del bloque 11


def test_impute_avg_price(datos_prueba):
    """Revisa que se rellenen los NAN con el promedio del precio del item correspondiente."""
    base, monthly = datos_prueba
    base = impute_avg_price(base, monthly)
    
    # Item 10 tiene promedio de precio 110
    assert base["avg_price"].iloc[0] == 110.0
    assert not base["avg_price"].isnull().any()