"""
Routines for polynomial fitting used for optimisation, IRC etc.
"""
from typing import Optional
import numpy as np


def _parabolic_fit():
    pass


def _get_poly_minimum(
    poly: np.poly1d,
    u_bound: Optional[float] = None,
    l_bound: Optional[float] = None,
) -> Optional[float]:
    """
    Obtain the minimum of a polynomial f(x), optionally within two
    bounds. If there are multiple minima, returns the lowest minimum

    Args:
        poly (np.poly1d): The polynomial whose minimum is requested
        u_bound (float|None): upper bound of range, optional
        l_bound (float|None): lower bound of range, optional

    Returns:
        (float): The position of the minimum x
    """
    # points with derivative = 0 are critical points
    crit_points = poly.deriv().roots

    if u_bound is not None:
        crit_points = crit_points[crit_points < u_bound]

    if l_bound is not None:
        crit_points = crit_points[crit_points > l_bound]

    if len(crit_points) == 0:
        return None

    minima = []
    # determine which are minima
    for point in crit_points:
        for i in range(2, 6):
            i_th_deriv = poly.deriv(i)(point)
            # if zero, move up another order of derivative
            if -1.0e-14 < i_th_deriv < 1.0e-14:
                continue
            # derivative > 0 and i is odd => minimum
            elif i_th_deriv > 0 and i % 2 != 0:
                minima.append(point)
            # otherwise maxima or inflection point
            else:
                break

    if len(minima) == 0:
        return None

    if len(minima) == 1:
        return minima[0]

    # more than one minima, have to check values
    minim_vals = [poly(x) for x in minima]

    return minima[np.argmin(minim_vals)]
