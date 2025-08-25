"""Reproducibility utilities for deterministic seeding and environment setup."""

import random
import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.utils import check_random_state
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic_torch: Whether to use deterministic PyTorch operations
                           (may reduce performance but ensures full reproducibility)
    """
    logger.info(f"Setting random seed to {seed}")
    
    # Standard library
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch if available
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        if deterministic_torch:
            # Enable deterministic operations (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # For newer PyTorch versions
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
    
    # Scikit-learn random state validation
    if HAS_SKLEARN:
        check_random_state(seed)
        
    logger.debug("Random seed configuration completed")


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """Get a numpy RandomState object with optional seed.
    
    Args:
        seed: Optional seed value. If None, uses current global state
        
    Returns:
        numpy.random.RandomState instance
    """
    return np.random.RandomState(seed)


def ensure_reproducible_environment() -> None:
    """Ensure environment is set up for reproducible results.
    
    This function checks and configures various environment settings
    that can affect reproducibility.
    """
    # Check PYTHONHASHSEED
    hash_seed = os.environ.get('PYTHONHASHSEED')
    if hash_seed != '0':
        logger.warning(
            f"PYTHONHASHSEED is '{hash_seed}', should be '0' for full reproducibility. "
            "Set environment variable: export PYTHONHASHSEED=0"
        )
    
    # Check for CUDA deterministic settings if PyTorch available
    if HAS_TORCH and torch.cuda.is_available():
        if not torch.backends.cudnn.deterministic:
            logger.warning("CUDA deterministic mode is disabled")
        if torch.backends.cudnn.benchmark:
            logger.warning("CUDA benchmark mode is enabled (may affect reproducibility)")
    
    logger.info("Reproducible environment check completed")


# Convenience function to set up full reproducible environment
def setup_reproducibility(seed: int = 42, check_env: bool = True) -> None:
    """Set up full reproducible environment with seed and environment checks.
    
    Args:
        seed: Random seed value
        check_env: Whether to check environment configuration
    """
    if check_env:
        ensure_reproducible_environment()
    
    set_seed(seed)