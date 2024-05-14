import numpy as np
from scipy.optimize import minimize
from typing import Callable
from functools import partial


def g1(x: float):
    return x


def g2(x: float):
    output = np.exp(x) if x <= 0 else 1 + x
    return output


def g2_antiderivative(x: float):
    output = np.exp(x) if x <= 0 else 1 + x + x**2 / 2
    return output


def loss(
    v: float,
    e: float,
    alpha: float,
    x: np.ndarray,
    g1: Callable[[float], float] = g1,
    g2: Callable[[float], float] = g2,
    g2_antiderivative: Callable[[float], float] = g2_antiderivative,
):
    """
    Equation 2 of https://arxiv.org/abs/1507.00244
    """
    indicator = x <= v
    loss_ = (
        (indicator - alpha) * (g1(v) - g1(x))
        + 1.0 / alpha * g2(e) * indicator * (v - x)
        + g2(e) * (e - v)
        - g2_antiderivative(e)
    )
    return loss_.mean()


def var_es(x: np.ndarray, alpha: float):
    """
    Minimisation from https://arxiv.org/abs/1507.00244
    """
    my_loss = lambda ve: loss(
        v=ve[0],
        e=ve[1],
        alpha=alpha,
        x=x,
        g1=g1,
        g2=lambda x: np.exp(x),
        g2_antiderivative=lambda x: np.exp(x),
    )  # noqa E731
    res = minimize(
        my_loss,
        x0=(0, 0),
    )
    return res
