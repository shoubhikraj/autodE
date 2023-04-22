"""
Routines for polynomial fitting used for optimisation, IRC etc.
"""
from typing import Optional, TYPE_CHECKING
import numpy as np


def two_point_cubic_fit(e0, g0, e1, g1):
    """
    Fit a general cubic equation with two points, using the energy
    and directional gradient at both points. Returns the fitted
    polynomial object. Equation: f(x) = ax**3 + bx**2 + cx + d

    Args:
        e0 (float):
        g0 (float):
        e1 (float):
        g1 (float):

    Returns:

    """
    # f(x) = ax**3 + bx**2 + cx + d
    # f(0) = d; f(1) = a + b + c + d
    d = e0
    # f'(x) = 3 a x**2 + 2 b x + c
    # f'(0) = c  => a+b = f(1) - c - d
    c = g0
    a_b = e1 - c - d
    # f'(1) = 3a + 2b + c => 3a+2b = f'(1) - c
    a3_2b = g1 - c
    a = a3_2b - 2 * a_b
    b = a_b - a
    return np.poly1d([a, b, c, d])


def two_point_exact_parabolic_fit(e0, g0, e1):
    """
    Fit a general parabolic (quadratic) equation with two points,
    using the energy and directional gradient at first point, and
    the energy at the second point. Returns the fitted polynomial.
    Equation: f(x) = a x**2  + b x + c

    Args:
        e0 (float): Energy at first point
        g0 (float): Directional (1D) gradient in search direction
        e1 (float): Energy at second point

    Returns:
        (np.poly1d): The fitted polynomial
    """
    # f(x) = ax**2 + bx + c
    # f(0) = c, f(1) = a + b + c => a + b = f(1) - f(0)
    c = e0
    a_b = e1 - c
    # f'(x) = 2ax + b; f'(0) = b
    # (a+b) - b = a
    b = g0
    a = a_b - b

    return np.poly1d([a, b, c])


def get_poly_minimum(
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
            # derivative > 0 and i is even => minimum
            elif i_th_deriv > 0 and i % 2 == 0:
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
