import numpy as np
import pytest

from scoring.metrics import gini


@pytest.mark.parametrize(
    ("p", "expected"),
    [([0.1, 0.2, 0.8, 0.9], 1.0), ([0.9, 0.8, 0.2, 0.1], -1.0), ([0.5, 0.5, 0.5, 0.5], 0.0)],
)
def test_gini_bounds(p: list[float], expected: float) -> None:
    assert gini(np.array([0, 0, 1, 1]), np.array(p)) == pytest.approx(expected)
