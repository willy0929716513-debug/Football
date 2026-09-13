import numpy as np

from football_predictor.regression import fit_logistic, fit_ridge_linear


def test_logistic_regression_recovers_feature_direction():
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(size=n)  # positively related to the outcome
    x2 = rng.normal(size=n)  # negatively related to the outcome
    z = 2.0 * x1 - 3.0 * x2
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < prob).astype(float)

    X = np.column_stack([x1, x2])
    model = fit_logistic(X, y, l2=1.0)

    assert model.coef[0] > 0
    assert model.coef[1] < 0


def test_logistic_regression_predict_proba_is_well_calibrated():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-1.5 * x))
    y = (rng.uniform(size=n) < prob).astype(float)

    model = fit_logistic(np.column_stack([x]), y, l2=0.5)
    predicted = model.predict_proba(np.column_stack([x]))
    # bucket by predicted probability and check realized frequency is close
    high_conf = predicted > 0.8
    assert high_conf.sum() > 50
    assert y[high_conf].mean() > 0.6


def test_ridge_linear_recovers_slope_and_intercept():
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(size=n)
    y = 4.0 + 2.5 * x + rng.normal(scale=0.1, size=n)

    model = fit_ridge_linear(np.column_stack([x]), y, l2=0.01)
    predicted = model.predict(np.column_stack([x]))
    assert np.mean(np.abs(predicted - y)) < 0.5
    # slope direction preserved regardless of internal standardization
    assert model.predict(np.array([[2.0]]))[0] > model.predict(np.array([[-2.0]]))[0]
