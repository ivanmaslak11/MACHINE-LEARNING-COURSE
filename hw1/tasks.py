import numpy as np


# 2 points
def euclidean_distance(X, Y) -> np.ndarray:
    """
    Compute element wise euclidean distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the Euclidean distance between the corresponding pair of vectors from the arrays X and Y
    """

    X_expanded = X[:, np.newaxis, :]
    Y_expanded = Y[np.newaxis, :, :]

    diff = (X_expanded - Y_expanded) ** 2

    return np.sqrt(np.sum(diff, axis=2))


# 2 points
def cosine_distance(X, Y) -> np.ndarray:
    """
    Compute element wise cosine distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the cosine distance between the corresponding pair of vectors from the arrays X and Y
    """
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    return 1 - X_normalized @ Y_normalized.T


# 1 point
def manhattan_distance(X, Y) -> np.ndarray:
    """
    Compute element wise manhattan distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the manhattan distance between the corresponding pair of vectors from the arrays X and Y
    """
    X_expanded = X[:, np.newaxis, :]
    Y_expanded = Y[np.newaxis, :, :]

    diff = np.abs(X_expanded - Y_expanded)

    return np.sum(diff, axis=2)
