"""
Base classes of second-order (quadratic) optimisers
"""
from abc import ABC
from typing import Optional, List

from autode.values import Distance
from autode.opt.optimisers.base import NDOptimiser


class QuadraticOptimiserBase(NDOptimiser, ABC):
    """Base class for quadratic optimisers that use trust radius"""

    def __init__(
        self,
        maxiter: int,
        conv_tol,
        init_hess: str,
        recalc_hess_every,
        init_trust: float,
        trust_update: bool,
        max_move,
        extra_prims,
        **kwargs,
    ):
        """ """
        super().__init__(maxiter=maxiter, conv_tol=conv_tol, **kwargs)

        self._trust = float(init_trust)
        assert self._trust > 0, "Trust radius has to be positive!"
        self._trust_update = bool(trust_update)
        self._maxmove = Distance(max_move, units="ang")
        assert self._maxmove > 0, "Max movement has to be positive!"
