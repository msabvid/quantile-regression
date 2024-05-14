from scipy.stats import norm
import numpy as np

from quantile_regression.expected_shortfall import var_es


def test_lognormal():
    """
    Test with explicit formula of negative of lognormal
    """
    mu = 0
    sigma = 1
    lambda_ = 0.5

    # see eq (18) of https://vega.xyz/papers/margins-and-credit-risk.pdf
    es_formula = -(
        np.exp(mu + sigma**2 / 2)
        / lambda_
        * (
            1
            - norm.cdf(
                norm.ppf(1 - lambda_, loc=mu, scale=sigma) - sigma, loc=mu, scale=sigma
            )
        )
    )

    rng = np.random.default_rng(seed=1)
    sample = -np.exp(rng.normal(loc=mu, scale=sigma, size=100000))
    res = var_es(sample, alpha=lambda_)
    assert abs(res.x[1] - es_formula) < 0.1
