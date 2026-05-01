import pytest
from unittest.mock import patch
from src.inference.inference import main

@patch("joblib.load")
@patch("src.inference.inference.setup_logger")
@patch("src.inference.inference.parse_args")
def test_main_model_not_found(mock_args, mock_logger, mock_load):
    """
    Prueba el manejo de errores para cuando no puede cargar el modelo.
    """
    # Hacemos que joblib arroje un FileNotFoundError al querer usar joblib.load
    mock_load.side_effect = FileNotFoundError("Model file not found")
    
    with pytest.raises(FileNotFoundError):
        main()
    
    # Segunda validación:
    # Verifica que sí se intentó cargar un modelo y que sólo se hizo una vez.
    mock_load.assert_called_once()