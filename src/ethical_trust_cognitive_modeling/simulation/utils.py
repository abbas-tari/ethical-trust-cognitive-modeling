# utils.py
import random

def set_seed(seed_value=52):
    """Set the seed for random number generators."""
    random.seed(seed_value)
    try:
        import numpy as np
        np.random.seed(seed_value)
    except ImportError:
        pass  # Numpy not installed
