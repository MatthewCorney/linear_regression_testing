from itertools import permutations, product
from math import factorial
from typing import Generator
import numpy as np


def safe_factorial(k: int) -> float:
    """
    Returns factorial(k) as a float, or +inf if it overflows.

    :param k: Nonnegative integer.
    :return: Factorial of k as float, or +inf on overflow.
    """
    try:
        return float(factorial(k))
    except OverflowError:
        return float('inf')


def psd_inv(M: np.ndarray, rcond=1e-12) -> np.ndarray:
    """
    Computes pseudoinverse of a (semi-)positive-definite matrix.

    :param M: Square matrix.
    :param rcond: Cutoff for small singular values.
    :return: Pseudoinverse of M.
    """
    return np.linalg.pinv(M, rcond=rcond)


def all_permutations(n: int) -> Generator[np.ndarray, None, None]:
    """
    Generates all permutations of [0..n-1].

    :param n: Size of the permutation range.
    :yield: Permutation as a (n,) NumPy array.
    """
    for perm in permutations(range(n)):
        yield np.array(perm, dtype=int)


def all_sign_flips(n: int) -> Generator[np.ndarray, None, None]:
    """
    Generates all sign-flip vectors of length n (±1 entries).

    :param n: Length of the sign-flip vector.
    :yield: (n,) NumPy array of ±1.
    """
    for signs in product([-1, 1], repeat=n):
        yield np.array(signs, dtype=int)
