"""Cache local de entrenamiento, predicciones y metricas para los notebooks."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone


CACHE_DIR = Path(__file__).resolve().parents[1] / "temp" / "forecast_cache"
memory = joblib.Memory(CACHE_DIR, verbose=0)
VERSIONS = (np.__version__, pd.__version__, sklearn.__version__, joblib.__version__)


@memory.cache
def _fit(model, X_train, y_train, versions):
    return clone(model).fit(X_train, y_train)


@memory.cache(ignore=["model", "X"])
def _predict(model, X, model_key, row_key, versions):
    return model.predict(X)


def fit_predict_cached(model, X_train, y_train, X_complete, config):
    """Invalidar entrenamiento al cambiar datos, parametros o dependencias."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "config": config,
        "model": repr(model),
        "parameters": repr(model.get_params(deep=True)),
        "versions": VERSIONS,
        "training_hash": joblib.hash((X_train, y_train)),
        "prediction_hash": joblib.hash(X_complete),
    }
    run_id = joblib.hash(record)
    (CACHE_DIR / f"{run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fitted = _fit(clone(model), X_train, y_train, VERSIONS)
    model_key = joblib.hash(fitted)
    # Bloques de una fila reutilizan predicciones anteriores al agregar datos.
    predictions = np.concatenate([
        _predict(
            fitted, X_complete.iloc[i:i + 1], model_key,
            (tuple(X_complete.columns), tuple(map(str, X_complete.dtypes)),
             tuple(X_complete.iloc[i].tolist())), VERSIONS,
        )
        for i in range(len(X_complete))
    ])
    return fitted, predictions
