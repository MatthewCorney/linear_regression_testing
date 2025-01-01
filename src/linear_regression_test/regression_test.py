import numpy as np
from typing import Literal, Optional, Union

from src.linear_regression_test.utils import safe_factorial, psd_inv, all_permutations, all_sign_flips


def randomize_responses(
        y: np.ndarray,
        method: Literal["perm", "flip", "both"],
        exact: bool,
        idx: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Given a response matrix/vector y (shape = (n, m)),
    applies one of the randomization methods:
      - 'perm': permutation
      - 'flip': sign flips
      - 'both': perm + flip
    If exact=True, uses the provided idx to do the exact reordering/sign flips.
    Otherwise, chooses them randomly.

    :param y: Response array of shape (n, m).
    :param method: 'perm' (permute rows), 'flip' (flip signs), or 'both'.
    :param exact: If True, use given idx for exact permutation/flips.
    :param idx: Indices/signs if exact=True.
    :return: Randomized copy of y.
    """

    n = y.shape[0]
    y_copy = y.copy()

    if method == 'perm':
        if exact:
            # idx is a permutation
            y_copy = y_copy[idx, :]
        else:
            perm_idx = np.random.permutation(n)
            y_copy = y_copy[perm_idx, :]

    elif method == 'flip':
        if exact:
            # idx is a sign vector
            y_copy = y_copy * idx[:, None]
        else:
            sign_vec = np.random.choice([-1, 1], size=n)
            y_copy = y_copy * sign_vec[:, None]

    else:  # method == 'both'
        if exact:
            # idx of length 2n: first n for permutation, second n for sign flips
            perm_part = idx[:n]
            flip_part = idx[n:]
            y_copy = y_copy[perm_part, :]
            y_copy = y_copy * flip_part[:, None]
        else:
            perm_idx = np.random.permutation(n)
            sign_vec = np.random.choice([-1, 1], size=n)
            y_copy = y_copy[perm_idx, :]
            y_copy = y_copy * sign_vec[:, None]

    return y_copy


def compute_tstat(
        x: np.ndarray,
        y: np.ndarray,
        xtx: Optional[np.ndarray],
        xtxi: Optional[np.ndarray],
        xinv: Optional[np.ndarray],
        ssy: Optional[np.ndarray],
        beta: Optional[np.ndarray],
        homosced: bool,
        lambda_used: bool
) -> Union[float, np.ndarray]:
    """
    Compute the test statistic for either:
      - Univariate Y (shape = (n, 1)) => returns a single float
      - Multivariate Y (shape = (n, m), m>1) => returns one T-value per column, OR
        if you only want the maximum across columns, you could do np.max(...) outside.

    If 'homosced' is True, uses the homoscedastic version;
    otherwise uses the heteroscedastic version.

    If beta is None, it is computed on the fly.  Similarly, for xtx, xtxi, xinv.

    :param x: Design matrix, shape (n, p).
    :param y: Response array, shape (n, m).
    :param xtx: X.T @ X or None.
    :param xtxi: (X.T @ X)^(-1) or pseudoinverse, or None.
    :param xinv: xtxi @ X.T or None.
    :param ssy: Sum of squares (if homoscedastic).
    :param beta: Coefficients or None.
    :param homosced: True for homoscedastic version.
    :param lambda_used: True if ridge penalty was used.
    :return: T-stat (float if univariate, array if multivariate).
    """

    # Make sure Y is 2D.
    if y.ndim == 1:
        y = y[:, None]
    nvar = y.shape[1]

    # Cross-products
    if xtx is None:
        xtx = x.T @ x
    if xtxi is None:
        xtxi = psd_inv(xtx)
    if xinv is None:
        xinv = xtxi @ x.T
    if beta is None:
        beta = xinv @ y

    # If univariate, just do the single-column approach
    if nvar == 1:
        return _compute_tstat_univ(x=x,
                                   y=y.squeeze(),
                                   xtx=xtx,
                                   xtxi=xtxi,
                                   ssy=ssy,
                                   beta=beta.squeeze(),
                                   homosced=homosced,
                                   lambda_used=lambda_used)

    # Otherwise do one T-value per column, then return them
    Tvals = np.zeros(nvar)
    for v in range(nvar):
        yv = y[:, v]
        bv = beta[:, v]
        ssy_v = ssy[v] if ssy is not None else None
        Tvals[v] = _compute_tstat_univ(x=x,
                                       y=yv,
                                       xtx=xtx,
                                       xtxi=xtxi,
                                       ssy=ssy_v,
                                       beta=bv,
                                       homosced=homosced,
                                       lambda_used=lambda_used)
    return Tvals


def _compute_tstat_univ(
        x: np.ndarray,
        y: np.ndarray,
        xtx: np.ndarray,
        xtxi: np.ndarray,
        ssy: Optional[float],
        beta: np.ndarray,
        homosced: bool,
        lambda_used: bool
) -> float:
    """
    Internal function for univariate T-statistic.

    :param x: Design matrix, shape (n, p).
    :param y: Response vector, shape (n,).
    :param xtx: X.T @ X.
    :param xtxi: Inverse or pseudoinverse of xtx.
    :param ssy: Sum of squares of y (if homoscedastic).
    :param beta: Estimated coefficients, shape (p,).
    :param homosced: True for homoscedastic formula.
    :param lambda_used: True if ridge penalty was used.
    :return: A T-statistic (float).
    """

    n, p = x.shape

    if homosced:
        # Homoscedastic version
        # rss = sum of squared residuals
        if lambda_used:
            # ssy - 2*(X'Y * beta) + sum( (X beta)^2 )
            rss = (ssy
                   - 2.0 * np.sum((x.T @ y) * beta)
                   + np.sum((x @ beta) ** 2))
        else:
            rss = ssy - np.sum((x.T @ y) * beta)

        sigma_hat = rss / (n - p - 1)
        Tstat = (beta.T @ (xtx @ beta)) / (p * sigma_hat)

    else:
        # Heteroscedastic version
        Z = x * np.abs(y)[:, None]  # shape (n, p)
        omega = Z.T @ Z
        middle = psd_inv(xtxi @ omega @ xtxi)
        Tstat = beta.T @ middle @ beta

    return Tstat


def perm_replication(
        x: np.ndarray,
        y: np.ndarray,
        method: Literal["perm", "flip", "both"],
        homosced: bool,
        exact: bool,
        xtx: np.ndarray,
        xtxi: np.ndarray,
        xinv: np.ndarray,
        ssy: Optional[np.ndarray],
        lambda_used: bool,
        idx: Optional[np.ndarray] = None
) -> float:
    """
    Performs one randomization replicate and returns the test statistic.

    :param x: Design matrix, shape (n, p).
    :param y: Response array, shape (n, m).
    :param method: Randomization type ('perm', 'flip', or 'both').
    :param homosced: True if homoscedastic formula is used.
    :param exact: True if an exact idx is provided.
    :param xtx: X.T @ X for x.
    :param xtxi: Pseudoinverse or inverse of xtx.
    :param xinv: xtxi @ x.T.
    :param ssy: Sum of squares (if homoscedastic).
    :param lambda_used: True if a ridge penalty is in use.
    :param idx: Permutation or sign-flip indices if exact=True.
    :return: Replicate test statistic (float).
    """

    # Randomize y (works for shape (n,1) or (n,m))
    Y_copy = randomize_responses(y, method, exact=exact, idx=idx)

    # Compute the T-values (multivariate => returns an array)
    Tvals = compute_tstat(
        x=x,
        y=Y_copy,
        xtx=xtx,
        xtxi=xtxi,
        xinv=xinv,
        ssy=ssy,
        beta=None,
        homosced=homosced,
        lambda_used=lambda_used
    )

    # If Tvals is array => combine by taking max
    if isinstance(Tvals, np.ndarray):
        return float(np.max(Tvals))
    else:
        return float(Tvals)



def subset_test(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        method: Literal["perm", "flip", "both"] = "perm",
        beta: Optional[np.ndarray] = None,
        homosced: bool = True,
        lambda_val: Union[float, np.ndarray] = 0.0,
        permutations: int = 9999,
        perm_dist: bool = True
) -> dict:
    """
    Freedman–Lane randomization test for X controlling for Z.

    :param x: Main predictors (n, p).
    :param y: Response(s) (n, m) or (n,).
    :param z: Nuisance/confounding variables (n, q).
    :param method: Randomization type. One of {"perm", "flip", "both"}.
    :param beta: Optional null hypothesis effect (p, m).
    :param homosced: True for homoscedastic T-stat.
    :param lambda_val: Ridge penalty (scalar or length p+q).
    :param permutations: Number of permutations if not enumerating exactly.
    :param perm_dist: If True, store permutation distribution in output.
    :return: Output dictionary
    """
    n, px = x.shape
    ny, m = y.shape
    nz, q = z.shape

    # If beta is specified, subtract it from y
    if beta is not None:
        y = y - x @ beta

    # 1) Construct the full design matrix W = [Z|X].  (size n x (q+px))
    W = np.hstack([z, x])
    p_full = q + px

    # 2) Fit the full model Y ~ Z + X => observed T
    #    Just adapt the same logic used in complete_test:
    #    (a) center W, center Y
    #    (b) incorporate lambda
    W_mean = W.mean(axis=0)
    y_mean = y.mean(axis=0)
    W_c = W - W_mean
    y_c = y - y_mean

    # Lambda array
    if np.isscalar(lambda_val):
        lambda_array = np.full(shape=(p_full,), fill_value=lambda_val, dtype=float)
    else:
        lambda_array = np.asarray(lambda_val, dtype=float)
        if lambda_array.size == 1:
            lambda_array = np.full(shape=(p_full,), fill_value=lambda_array[0])
        if (lambda_array < 0).any():
            raise ValueError("lambda_val must be non-negative.")

    use_lambda = np.any(lambda_array > 0)

    # Cross products
    WtW = W_c.T @ W_c + np.diag(n * lambda_array)  # (p_full x p_full)
    WtW_inv = psd_inv(WtW)
    W_inv = WtW_inv @ W_c.T  # (p_full x n)

    # Fit coefficients => shape (p_full x m)
    coefs_full = W_inv @ y_c

    # Possibly store sum of squares if homosced
    if homosced:
        ssy = np.sum(y_c ** 2, axis=0)  # shape (m,)
    else:
        ssy = None

    # Observed T
    Tvals = compute_tstat(
        x=W_c,
        y=y_c,
        xtx=WtW,
        xtxi=WtW_inv,
        xinv=W_inv,
        ssy=ssy,
        beta=coefs_full,
        homosced=homosced,
        lambda_used=use_lambda
    )
    if isinstance(Tvals, float):
        obs_T = Tvals
        Tuni = np.array([obs_T])
    else:
        obs_T = np.max(Tvals)
        Tuni = Tvals

    # 3) Freedman–Lane randomization: partial out Z from Y
    #    That means we do a ridge-regression of Y on Z alone.
    #    Then R(Y) = Y - fittedZ
    #    We'll do the same "center, then penalize" logic for Z alone:
    Z_c = z - z.mean(axis=0)
    # penalty for Z part: same approach but only for the first q positions?
    # Typically Freedman–Lane requires we use the same approach for partialing
    # as for the final model. We'll incorporate a q-dimensional slice for lambda_z:
    if q > 0:
        lambda_z = lambda_array[:q]  # first q entries for the Z columns
    else:
        lambda_z = np.array([])  # trivial case

    ZtZ = Z_c.T @ Z_c + np.diag(n * lambda_z) if q > 0 else np.zeros((0, 0))
    if q > 0:
        ZtZ_inv = psd_inv(ZtZ)
        Z_inv = ZtZ_inv @ Z_c.T
        # fittedZ_c = Z_c @ coefsZ  where coefsZ = Z_inv @ y_c
        coefsZ = Z_inv @ y_c  # shape (q x m)
        fittedZ_c = Z_c @ coefsZ
    else:
        # No actual Z => no partialing out
        fittedZ_c = np.zeros_like(y_c)

    # So the residual from partialing out Z is:
    RY_c = y_c - fittedZ_c  # shape (n, m)

    # 4) Count possible permutations => exact or approximate?
    if method == 'perm':
        total_possible = safe_factorial(n)
    elif method == 'flip':
        total_possible = 2 ** n if n <= 60 else float('inf')
    else:  # 'both'
        fac_n = safe_factorial(n)
        if fac_n == float('inf'):
            total_possible = float('inf')
        else:
            two_pow_n = 2 ** n if n <= 60 else float('inf')
            total_possible = fac_n * two_pow_n

    exact = (total_possible <= (permutations + 1))

    # 5) Build the randomization distribution by Freedman–Lane:
    #    For each randomization:
    #       R*(Y)_c = randomize_responses(RY_c, method)
    #       Y*_c = fittedZ_c + R*(Y)_c
    #       Fit Y*_c ~ W_c => T*
    perm_values = []
    if permutations == 0:
        # no permutations => just store None
        perm_dist_array = None
        p_value = None
        used_permutations = 0

    else:
        if exact:
            # Enumerate all permutations, flips, or both
            if method == 'perm':
                for perm_idx in all_permutations(n):
                    RY_star_c = randomize_responses(RY_c, 'perm', exact=True, idx=perm_idx)
                    Y_star_c = fittedZ_c + RY_star_c
                    T_star = compute_tstat(
                        x=W_c, y=Y_star_c,
                        xtx=WtW, xtxi=WtW_inv, xinv=W_inv,
                        ssy=ssy, beta=None,
                        homosced=homosced, lambda_used=use_lambda
                    )
                    if isinstance(T_star, np.ndarray):
                        perm_values.append(np.max(T_star))
                    else:
                        perm_values.append(T_star)

            elif method == 'flip':
                for flip_vec in all_sign_flips(n):
                    RY_star_c = randomize_responses(RY_c, 'flip', exact=True, idx=flip_vec)
                    Y_star_c = fittedZ_c + RY_star_c
                    T_star = compute_tstat(
                        x=W_c, y=Y_star_c,
                        xtx=WtW, xtxi=WtW_inv, xinv=W_inv,
                        ssy=ssy, beta=None,
                        homosced=homosced, lambda_used=use_lambda
                    )
                    if isinstance(T_star, np.ndarray):
                        perm_values.append(np.max(T_star))
                    else:
                        perm_values.append(T_star)

            else:  # 'both'
                all_perms = list(all_permutations(n))
                all_flips = list(all_sign_flips(n))
                for perm_idx in all_perms:
                    for flip_vec in all_flips:
                        combo_idx = np.concatenate([perm_idx, flip_vec])
                        RY_star_c = randomize_responses(RY_c, 'both', exact=True, idx=combo_idx)
                        Y_star_c = fittedZ_c + RY_star_c
                        T_star = compute_tstat(
                            x=W_c, y=Y_star_c,
                            xtx=WtW, xtxi=WtW_inv, xinv=W_inv,
                            ssy=ssy, beta=None,
                            homosced=homosced, lambda_used=use_lambda
                        )
                        if isinstance(T_star, np.ndarray):
                            perm_values.append(np.max(T_star))
                        else:
                            perm_values.append(T_star)

            perm_dist_array = np.array(perm_values, dtype=float)
            used_permutations = len(perm_dist_array)

        else:
            # approximate
            perm_dist_array = np.zeros(permutations)
            for i in range(permutations):
                RY_star_c = randomize_responses(RY_c, method, exact=False)
                Y_star_c = fittedZ_c + RY_star_c
                T_star = compute_tstat(
                    x=W_c, y=Y_star_c,
                    xtx=WtW, xtxi=WtW_inv, xinv=W_inv,
                    ssy=ssy, beta=None,
                    homosced=homosced, lambda_used=use_lambda
                )
                if isinstance(T_star, np.ndarray):
                    perm_dist_array[i] = np.max(T_star)
                else:
                    perm_dist_array[i] = T_star
            used_permutations = permutations

        p_value = np.mean(perm_dist_array >= obs_T)

    # 6) Reconstruct intercept + slopes for final [Z|X] => Y
    #    Because we centered W, Y, the intercept for each response dim is:
    #      alpha = (original Y-mean) - (W_mean) @ coefs_full
    coefs_corrected = coefs_full
    alpha = y_mean - W_mean @ coefs_corrected
    # shape => (p_full+1, m)
    final_coefs = np.vstack([alpha, coefs_corrected])

    # 7) Package output
    result = {
        'statistic': obs_T,
        'p_value': p_value,
        'perm_dist': perm_dist_array if perm_dist else None,
        'method': f"Freedman-Lane ({method})",
        'null_value': beta if beta is not None else None,
        'homosced': homosced,
        'R': used_permutations,
        'exact': exact,
        'coefficients': final_coefs
    }
    if m > 1:
        result['univariate'] = Tuni
        if perm_dist_array is not None:
            # compare each dimension's T-value vs the distribution of max-stats
            adj_pvals = [np.mean(perm_dist_array >= tval) for tval in Tuni]
            result['adj_p_values'] = np.array(adj_pvals)
        else:
            result['adj_p_values'] = None

    return result


def complete_test(x: np.ndarray,
                  y: np.ndarray,
                  lambda_val: Union[float, np.ndarray] = 0.0,
                  method: Literal["perm", "flip", "both"] = "perm",
                  beta: Optional[np.ndarray] = None,
                  homosced: bool = True,
                  permutations: int = 9999,
                  perm_dist: bool = True):
    """
    Randomization test (univariate or multivariate) with optional ridge penalty.

    :param x: Design matrix (n, p).
    :param y: Response(s) (n, m) or (n,).
    :param n: Number of observations (same as x.shape[0]).
    :param lambda_val: Ridge penalty (scalar or array-like).
    :param method: Randomization type.
    :param beta: Optional null hypothesis effect (p, m).
    :param homosced: True for homoscedastic T-stat.
    :param permutations: Number of permutations or flips if not enumerating exactly.
    :param perm_dist: If True, store permutation distribution.
    :return: output dictionary
    """
    # Center x, y
    n, p = x.shape
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Process lambda_val
    if np.isscalar(lambda_val):
        lambda_array = np.full(shape=(p,), fill_value=lambda_val, dtype=float)
    else:
        lambda_val = np.asarray(lambda_val, dtype=float)
        if (lambda_val < 0).any():
            raise ValueError("lambda_val must be non-negative.")
        if lambda_val.size == p:
            lambda_array = lambda_val
        else:
            # fallback: repeat first element
            lambda_array = np.full(shape=(p,), fill_value=lambda_val[0])

    # Are we actually using the penalty?
    use_lambda = np.any(lambda_array > 0)
    # Compute cross products
    xtx = x_centered.T @ x_centered + np.diag(n * lambda_array)
    xtxi = psd_inv(xtx)
    xinv = xtxi @ x_centered.T

    # Fit coefficients on the centered data
    # shape coefs: (p, m)
    coefs = xinv @ y_centered

    # If homosced, store sum of squares
    ssy = None
    if homosced:
        ssy = np.sum(y_centered ** 2, axis=0)  # shape (m,)

    # 1) Observed statistic
    Tvals = compute_tstat(
        x_centered,
        y_centered,
        xtx,
        xtxi,
        xinv,
        ssy,
        coefs,
        homosced=homosced,
        lambda_used=use_lambda
    )
    if isinstance(Tvals, float):
        # univariate => single float
        obs_T = Tvals
        Tuni = np.array([obs_T])
    else:
        # multivariate => array of shape (m,)
        Tuni = Tvals
        obs_T = np.max(Tuni)

    # 2) Exact vs approximate permutation count
    #    method='perm' => n! permutations
    #    method='flip' => 2^n flips
    #    method='both' => n! * 2^n
    # We check if that count <= permutations + 1 => exact

    if method == 'perm':
        total_possible = safe_factorial(n)
    elif method == 'flip':
        total_possible = 2 ** n if n <= 60 else float('inf')  # watch out for large n
    else:  # 'both'
        fac_n = safe_factorial(n)
        if fac_n == float('inf'):
            total_possible = float('inf')
        else:
            # multiply by 2^n carefully
            two_pow_n = 2 ** n if n <= 60 else float('inf')
            total_possible = fac_n * two_pow_n

    exact = (total_possible <= (permutations + 1))

    # 3) Build the permutation distribution
    if permutations == 0:
        # No permutations => parametric fallback or skip
        perm_dist_array = None
        p_value = None
        used_permutations = 0

    else:
        if exact:
            # Enumerate all permutations / flips / both
            perm_values = []
            if method == 'perm':
                for perm_idx in all_permutations(n):
                    val = perm_replication(
                        x_centered, y_centered,
                        method='perm', homosced=homosced, exact=True,
                        xtx=xtx, xtxi=xtxi, xinv=xinv, ssy=ssy,
                        lambda_used=use_lambda, idx=perm_idx
                    )
                    perm_values.append(val)

            elif method == 'flip':
                for flip_vec in all_sign_flips(n):
                    val = perm_replication(
                        x_centered, y_centered,
                        method='flip', homosced=homosced, exact=True,
                        xtx=xtx, xtxi=xtxi, xinv=xinv, ssy=ssy,
                        lambda_used=use_lambda, idx=flip_vec
                    )
                    perm_values.append(val)

            else:  # 'both'
                # cross over permutations x flips
                all_perms = list(all_permutations(n))
                all_flips = list(all_sign_flips(n))
                for perm_idx in all_perms:
                    for flip_vec in all_flips:
                        combined_idx = np.concatenate([perm_idx, flip_vec])
                        val = perm_replication(
                            x_centered, y_centered,
                            method='both', homosced=homosced, exact=True,
                            xtx=xtx, xtxi=xtxi, xinv=xinv, ssy=ssy,
                            lambda_used=use_lambda, idx=combined_idx
                        )
                        perm_values.append(val)

            perm_dist_array = np.array(perm_values, dtype=float)
            used_permutations = len(perm_dist_array)

        else:
            # Approximate
            perm_values = np.zeros(permutations)
            for i in range(permutations):
                perm_values[i] = perm_replication(
                    x_centered, y_centered,
                    method=method, homosced=homosced, exact=False,
                    xtx=xtx, xtxi=xtxi, xinv=xinv, ssy=ssy,
                    lambda_used=use_lambda
                )
            perm_dist_array = np.concatenate(([obs_T], perm_values))
            used_permutations = len(perm_dist_array)

        # Compute p-value: proportion of permuted stats >= obs_T
        p_value = np.mean(perm_dist_array >= obs_T)

    # 4) Reconstruct intercept + slopes for final coefficients
    #    Because we centered x and y,
    #    alpha (intercept) = (original y-mean) - x_mean @ (slopes)
    #    Also if a null beta was subtracted from y, we add it back now.

    coefs_corrected = coefs.copy()
    y_means_corrected = y_mean.copy()
    if beta is not None:
        # y_mean was shifted by x_mean @ beta
        # so we add that back in
        y_means_corrected += x_mean @ beta
        coefs_corrected += beta

        # intercept for each response dimension
    alpha = y_means_corrected - x_mean @ coefs_corrected
    # Final shape => (p+1, m)
    final_coefs = np.vstack([alpha, coefs_corrected])

    # Prepare output
    output = {
        'statistic': obs_T,
        'p_value': p_value,
        'perm_dist': perm_dist_array if perm_dist else None,
        'method': method,
        'null_value': beta if beta is not None else None,
        'homosced': homosced,
        # If enumerating exactly, the first replicate is not included in the permutations
        # in some definitions. Here we just store how many total draws were done.
        'R': used_permutations,
        'exact': exact,
        'coefficients': final_coefs
    }

    # If multivariate, store per-column T-values and adjusted p-values
    m = y.shape[1]
    if m > 1:
        output['univariate'] = Tuni
        if perm_dist_array is not None:
            # Conservative approach: compare each Tuni[v] to the distribution of max-stats
            # => adjusted p-values
            adj_pvals = [np.mean(perm_dist_array >= tval) for tval in Tuni]
            output['adj_p_values'] = np.array(adj_pvals)
        else:
            output['adj_p_values'] = None
    return output


def regularisation_test(
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None,
        permutation_method: Literal["perm", "flip", "both"] = "perm",
        beta: Optional[np.ndarray] = None,
        homosced: bool = True,
        lambda_val: Union[float, np.ndarray] = 0.0,
        permutations: int = 9999,
        perm_dist: bool = True
) -> dict:
    """
    Randomization test for regression with optional ridge penalty and confounders.

    :param x: Design matrix (n, p).
    :param y: Response(s) (n,) or (n, m).
    :param z: Optional confounders (n, q).
    :param permutation_method: Randomization type ("perm", "flip", "both").
    :param beta: Optional null-hypothesis coefficients (p, m).
    :param homosced: True for homoscedastic formula.
    :param lambda_val: Ridge penalty (scalar or array-like).
    :param permutations: Number of permutations if not exact enumeration.
    :param perm_dist: If True, include permutation distribution in the output.
    :return: A dictionary with:
            'statistic': observed test statistic
            'p_value': randomization p-value
            'perm_dist': the permutation distribution (or None)
            'method': the method used
            'null_value': the specified beta (or None)
            'homosced': whether homoscedastic assumption was used
            'R': the number of permutations actually used
            'exact': flag indicating whether an exhaustive enumeration was performed
            'coefficients': final fitted coefficients (intercept + slopes)
            Additional keys for multivariate responses:
                'univariate': array of T-statistics for each response dimension
                'adj_p_values': conservative p-values for each dimension
    """
    # Convert to array
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if z is not None:
        z = np.asarray(z, dtype=float)
    n, p = x.shape
    # Ensure y is 2D
    if y.ndim == 1:
        y = y[:, None]

    # check x shape
    ny, m = y.shape
    if ny != n:
        raise ValueError("x and y must have same # of rows")

    # Check z shape
    if z is not None:
        nz, q = z.shape
        if nz != n:
            raise ValueError("z must have same # of rows as x, y")

    # Check beta shape
    if beta is not None:
        beta = np.asarray(beta, dtype=float)
        if beta.shape[0] != p:
            raise ValueError("beta has incompatible shape (rows != p).")
        if beta.shape[1] != y.shape[1]:
            raise ValueError("beta has incompatible shape (cols != number of responses).")

    # Validate method
    valid_methods = ['perm', 'flip', 'both']
    if permutation_method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Subtract the null-hypothesis effect if beta is provided
    if beta is not None:
        y = y - x @ beta
    if z is not None:
        output = subset_test(x=x,
                             y=y,
                             z=z,
                             method=permutation_method,
                             homosced=homosced,
                             beta=beta,
                             permutations=permutations,
                             perm_dist=perm_dist,
                             lambda_val=lambda_val,
                             )
        return output
    else:

        output = complete_test(x=x,
                               y=y,
                               method=permutation_method,
                               homosced=homosced,
                               beta=beta,
                               permutations=permutations,
                               perm_dist=perm_dist,
                               lambda_val=lambda_val,
                               )
        return output
