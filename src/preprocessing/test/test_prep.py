import pytest
import pandas as pd
from src.preprocessing.prep import validate_columns


def test_validate_columns_exito():
    """Prueba que no se levanten excepciones cuando sí están las columnas requeridas."""
    df = pd.DataFrame({"columna_correcta": [1], "columna_extra": [2]})
    
    # No debería levantar excepciones
    validate_columns(df, ["columna_correcta"], "archivo.csv")


def test_validate_columns_error():
    """Prueba que se registre un error si faltan columnas requeridas."""
    df = pd.DataFrame({"columna_incorrecta_1": [1], "columna_incorrecta_2": [2]})
    required = ["columna_requerida"]

    # Debería levantar la excepción ValueError
    with pytest.raises(ValueError, match="Missing columns in archivo.csv"):
        validate_columns(df, required, "archivo.csv")