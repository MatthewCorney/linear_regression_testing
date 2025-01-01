import pytest
import pandas as pd
import numpy as np
from src.linear_regression_test.regression_test import regularisation_test

R_RESULTS = pd.read_csv("testing_data/R_results.csv")



@pytest.mark.parametrize("row", R_RESULTS.to_dict('records'))
def test_regularisation_test(row):
    """
    Parametrized test that checks p-value and statistic from R vs Python's regularisation_test.
    """
    target_p_val = row['p_value']
    target_statistic = row["statistic"]
    homosced_value = row['homosced']
    method_value = row['method']
    complete_bool = row['complete']
    variate_type = row['variate']
    number_items = row['sample']
    if not complete_bool:
        pytest.skip("Skipping because method is not FL for incomplete data.")
        if method_value != "FL":
            pytest.skip("Skipping because method is not FL for incomplete data.")
        else:
            data = pd.read_csv(f"testing_data/{variate_type}variate_n{number_items}.csv")
            x_test = data[[col for col in data.columns if col.startswith('x1')]].to_numpy()
            y_test = data[[col for col in data.columns if col.startswith('y')]].to_numpy()
            z_test = data[[col for col in data.columns if col.startswith('x2')]].to_numpy()

            res_m = regularisation_test(
                x_test,
                y_test,
                z_test,
                permutation_method='perm',
                beta=None,
                homosced=homosced_value,
                lambda_val=0,
                permutations=9999,
                perm_dist=True
            )
            assert np.allclose(target_p_val, res_m['p_value'], atol=0.1), (
                f"\n[FL incomplete] p-value mismatch\n"
                f"Expected (R) = {target_p_val:.3f}\nGot (Python) = {res_m['p_value']:.3f}"
            )
            assert np.allclose(target_statistic, res_m['statistic'], atol=0.1), (
                f"\n[FL incomplete] statistic mismatch\n"
                f"Expected (R) = {target_statistic:.3f}\nGot (Python) = {res_m['statistic']:.3f}"
            )

    else:
        data = pd.read_csv(f"testing_data/{variate_type}variate_n{number_items}.csv")
        x_test = data[[col for col in data.columns if col.startswith('x')]].to_numpy()
        y_test = data[[col for col in data.columns if col.startswith('y')]].to_numpy()

        res = regularisation_test(
            x_test,
            y_test,
            permutation_method=method_value,
            beta=None,
            homosced=homosced_value,
            lambda_val=0,
            permutations=9999,
            perm_dist=True
        )

        assert np.allclose(target_p_val, res['p_value'], atol=0.1), (
            f"\n[complete] p-value mismatch\n"
            f"Expected (R) = {target_p_val:.3f}\nGot (Python) = {res['p_value']:.3f}"
        )
        assert np.allclose(target_statistic, res['statistic'], atol=0.1), (
            f"\n[complete] statistic mismatch\n"
            f"Expected (R) = {target_statistic:.3f}\nGot (Python) = {res['statistic']:.3f}"
        )
