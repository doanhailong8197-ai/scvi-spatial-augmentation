"""
Source code repository for the scVI-balanced generative augmentation manuscript.
Contains modules for statistical evaluation, standardized visualization, 
and performance metrics computation.
"""

from . import stats
from . import plotting
from . import evaluation

__all__ = [
    "stats",
    "plotting",
    "evaluation"
]