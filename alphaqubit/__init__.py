"""AlphaQubit: an open reference implementation of a recurrent-transformer
neural decoder for surface codes.

See the README for the mapping to Bausch et al., Nature 636, 834 (2024).
"""
from .model import AlphaQubit, AlphaQubitConfig
from .data import CodeTask, make_task
from .evaluate import model_logical_error_rate

__all__ = [
    "AlphaQubit",
    "AlphaQubitConfig",
    "CodeTask",
    "make_task",
    "model_logical_error_rate",
]

__version__ = "0.1.0"
