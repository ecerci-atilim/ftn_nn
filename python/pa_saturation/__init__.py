"""
FTN with PA Saturation - Python Package

Power Amplifier saturation nonlinearity models and FTN simulation

Author: Emre Cerci
Date: January 2026
"""

from .pa_models import (
    rapp_model,
    saleh_model,
    soft_limiter,
    apply_pa_model,
    get_pa_params_from_ibo
)

__all__ = [
    'rapp_model',
    'saleh_model',
    'soft_limiter',
    'apply_pa_model',
    'get_pa_params_from_ibo'
]

__version__ = '1.0.0'
__author__ = 'Emre Cerci'
