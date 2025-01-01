
## Introduction
This repository provides an implementation of various randomization-based statistical tests, including the Freedman–Lane procedure for regression scenarios with optional confounding variables and ridge-type regularization. The core functionality allows you to:

- Compute univariate or multivariate T-statistics for linear models.
- Perform randomization tests with permutations, sign flips, or both.
- Optionally partial out confounders (Freedman–Lane approach).
- Apply ridge regularization when fitting models.
- Conduct exhaustive enumeration of permutations when feasible, or fallback to approximate random sampling otherwise.

## Dependencies
- numpy  
- pandas  

## Usage
The tests are all handled by `regularisation_test`


- **`z`:** partiales out `Z` if provided.
- **`method`:** One of `"perm"`, `"flip"`, or `"both"`.  
- **`beta`:** Optional null hypothesis vector/matrix to subtract from `y`.  
- **`lambda_val`:** Ridge penalty (float or array).  
- **`permutations`:** Number of random permutations if exact enumeration is not used.


### Important Parameters

- **`method`:**  
  - `"perm"`: Randomly permute row indices.  
  - `"flip"`: Randomly flip signs (+1/−1).  
  - `"both"`: First permute, then flip.

- **`beta`:**  
  A hypothesis-specific set of coefficients to remove from `y` before the test.  
  Shape must match `(p, m)` where `p` is number of predictors, and `m` is number of response variables (if multivariate).

- **`lambda_val`:**  
  - Accepts a scalar or array.  
  - When scalar, repeated across all columns of the design.  
  - Applied as a ridge-type penalty in the matrix inverse step.

- **`homosced`:**  
  - If `True`, uses homoscedastic T-stat (residuals assumed to have constant variance).  
  - If `False`, uses a heteroscedastic formula (more robust at cost of complexity).

- **`permutations`:**  
  - Number of random permutations/flips.  
  - If exact enumeration is smaller or equal to `permutations + 1`, the code enumerates all possibilities.

## Example
```
pip install git+https://github.com/MatthewCorney/linear_regression_testing.git
```

```
x = np.array([[2.07096901, - 1.16749027],
              [-0.04043207, 0.62415002],
              [1.73666627, - 0.04823333],
              [-0.04770574, - 0.34957507],
              [-1.3890172, - 0.08557927],
              [-0.52603589, 0.28846998],
              [-1.52849969, - 0.35389813],
              [0.01474273, 0.67071754],
              [-0.57194783, - 0.85265874],
              [-0.88949808, 1.43062724]])

y = np.array([[0.41302607],
              [-0.59289967],
              [-1.74087555],
              [0.18168425],
              [1.61111585],
              [-0.77134736],
              [-1.10398284],
              [-0.2462571],
              [-0.95332138],
              [-1.88845488]])
res = regularisation_test(
    x,
    y,
    permutation_method='perm',
    beta=None,
    homosced=False,
    lambda_val=0,
    permutations=9999,
    perm_dist=True
)
print(f"P-Value: {res['p_value']: .3f}")
print(f"Statistic: {res['statistic']: .3f}")
```

```
P-Value: 0.521
Statistic: 1.849
```