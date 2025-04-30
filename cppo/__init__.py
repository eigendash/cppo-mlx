"""A small MLX implementation of GRPO with CPPO-style completion pruning.

The paper is *CPPO: Accelerating the Training of Group Relative Policy
Optimization-Based Reasoning Models* (arXiv:2503.22342).  This package holds a
tiny causal LM, a programmatic-reward synthetic task, the GRPO objective and the
CPPO pruning / dynamic-allocation layer on top of it.
"""

__version__ = "0.1.0"

__all__: list[str] = []
