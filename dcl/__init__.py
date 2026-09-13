"""Decision-Censored Learning.

Evaluation and learning when the label is observed only where an incumbent
policy decided to look.  See the README for the sensitivity model DCSM(Gamma)
and the guarantees each module implements.
"""

from .auc_bounds import (AUROC_SPEC, MetricSpec, auc_direct, auc_from_regression,
                         midrank, naive_corner_interval, separate_bounds_interval,
                         sharp_auc_interval, sharp_bounds)
from .evaluation import bounds_report, deployment_metrics, observed_metrics
from .falsify import falsification_curve, gamma_lower_bound
from .models import DCLParametric, DCLPlugin
from .nuisance import CrossFitNuisance, NuisanceEstimates
from .objectives import (abstention_band, dcl_bayes_score, least_favourable_p,
                         minimax_regret, worstcase_risk,
                         worstcase_risk_decomposition)
from .ranking import DCLRanker, worst_case_auc
from .sensitivity import (IdentifiedSet, expit, identified_set, logit,
                          outcome_bounds, realised_gamma)
from .uq import naive_decomposition, three_way_decomposition

__version__ = "0.1.0"

__all__ = [
    # sensitivity model
    "IdentifiedSet", "identified_set", "outcome_bounds", "realised_gamma",
    "expit", "logit",
    # ranking bounds (Theorem 2)
    "midrank", "auc_from_regression", "auc_direct", "sharp_auc_interval",
    "sharp_bounds", "MetricSpec", "AUROC_SPEC", "naive_corner_interval",
    "separate_bounds_interval",
    # learning (Theorem 6 / Proposition 7)
    "worstcase_risk", "worstcase_risk_decomposition", "dcl_bayes_score",
    "minimax_regret", "least_favourable_p", "abstention_band",
    "DCLPlugin", "DCLParametric", "DCLRanker", "worst_case_auc",
    # nuisances and evaluation
    "CrossFitNuisance", "NuisanceEstimates", "deployment_metrics",
    "observed_metrics", "bounds_report",
    # falsification and uncertainty
    "gamma_lower_bound", "falsification_curve",
    "naive_decomposition", "three_way_decomposition",
    "__version__",
]
