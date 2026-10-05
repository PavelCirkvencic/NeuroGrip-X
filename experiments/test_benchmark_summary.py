"""Regression tests for scenario-macro aggregation in the benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark_summary import aggregate_scenario_rmse, macro_average_scenarios


def test_runs_of_same_scenario_are_pooled_not_overwritten():
    """Two runs of scenario A must pool their residuals, not overwrite."""
    small = np.array([[0.0, 0.01], [0.0, 0.01]])
    large = np.array([[0.0, 0.11], [0.0, 0.11]])
    result = aggregate_scenario_rmse({"A": [small, large]})
    pooled = np.sqrt(np.mean(np.concatenate([small, large])[:, 1] ** 2))
    last_run_only = np.sqrt(np.mean(large[:, 1] ** 2))
    assert result["A"] == pytest.approx(pooled)
    # The old overwrite behaviour would have returned `last_run_only`.
    assert result["A"] != pytest.approx(last_run_only)


def test_macro_weights_scenarios_equally_not_runs():
    """The macro average gives each scenario weight 1, regardless of runs."""
    per_scenario = {"A": 1.0, "B": 3.0}
    assert macro_average_scenarios(per_scenario) == pytest.approx(2.0)
    # A micro average over three runs of B and one of A would be 2.5 instead.
    micro = (1.0 + 3.0 + 3.0 + 3.0) / 4.0
    assert macro_average_scenarios(per_scenario) != pytest.approx(micro)


def test_empty_input_is_nan():
    """An empty set of scenarios yields NaN instead of raising."""
    assert np.isnan(macro_average_scenarios({}))
    assert aggregate_scenario_rmse({}) == {}
