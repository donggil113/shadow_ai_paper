"""Selective-labels datasets used in the paper."""

from .base import SelectiveLabelsDataset, standardise
from .synthetic import make_sl_bench, oracle_quantities
from .semisynthetic import censor_with_hidden_signal, residualise
from .lending import make_lending_club
from .compas import make_compas
from .creditcard import make_creditcard
from .mimic import make_mimic, make_mimic_sim

REGISTRY = {
    "sl_bench": make_sl_bench,
    "lending_club": make_lending_club,
    "mimic_sim": make_mimic_sim,
    "mimic": make_mimic,
    "compas": make_compas,
    "creditcard": make_creditcard,
}


def load(name: str, **kwargs) -> SelectiveLabelsDataset:
    """Load a dataset by registry name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["SelectiveLabelsDataset", "standardise", "make_sl_bench",
           "oracle_quantities", "censor_with_hidden_signal", "residualise",
           "make_lending_club", "make_compas", "make_creditcard",
           "make_mimic", "make_mimic_sim", "REGISTRY", "load"]
