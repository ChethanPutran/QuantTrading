import numpy as np

from models.linear import LinearModel


def test_linear_model_predict_and_update():
    lm = LinearModel(n_features=3, learning_rate=0.1, l2=0.0, fit_intercept=True)
    x = np.array([1.0, 2.0, 3.0])
    # initial predict should be zero
    pred0 = lm.predict(x)
    assert abs(pred0) < 1e-8

    # update towards y=1.0
    lm.update(x, 1.0)
    pred1 = lm.predict(x)
    assert isinstance(pred1, float)
    assert pred1 != pred0
