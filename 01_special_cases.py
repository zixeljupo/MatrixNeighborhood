import numpy as np
from itertools import combinations, product, chain


# checks that matrix is int
def is_matrix_integer(matrix):
    return matrix.dtype == int


# checks that matrix contains only zeros and ones
def is_matrix_binary(matrix):
    return matrix.min() >= 0 and matrix.max() <= 1


# checks that matrix has correct structure
def is_matrix_correct(matrix):
    return max(matrix.sum(axis=1)) <= 1


# set definition
class Ank:

    def get_value(self):
        return self._a.copy()

    def get_n(self):
        return self._n

    def get_k(self):
        return self._k

    # constructor with checks
    def __init__(self, source_matrix: np.matrix):

        if not is_matrix_integer(source_matrix):
            raise ValueError("Current TaskMatrix realization works only with int matrices")

        if not is_matrix_binary(source_matrix):
            raise ValueError("Passed matrix is not binary")

        if not is_matrix_correct(source_matrix):
            raise ValueError("Passed matrix is not under task rules")

        self._a = source_matrix
        self._n = source_matrix.shape[0]
        self._k = source_matrix.shape[1]


# calculates Hamming distance between two task matrices
def hamming_distance(a: Ank, b: Ank):
    return np.not_equal(a.get_value(), b.get_value()).astype(int).sum()


def get_ones_positions(matrix):
    has_value = matrix.sum(axis=1).astype(bool)
    values_positions = np.where(matrix == 1)

    where_value = np.full_like(has_value, -1, dtype=int)
    np.put(where_value, values_positions[0], values_positions[1])

    return has_value, where_value


def generator(a, n, k, D):
    has_ones, where_ones = get_ones_positions(a)
    ones_count = has_ones.astype(int).sum()

    for d in D:
        if d == 0:
            yield Ank(a.copy())
        if d == 1:
            for i in range(n):
                if has_ones[i]:
                    _matrix = a.copy()
                    _matrix[i] = 0
                    yield Ank(_matrix)
                else:
                    for j in range(k):
                        _matrix = a.copy()
                        _matrix[i, j] = 1
                        yield Ank(_matrix)
        if d == 2:

            for rows in combinations(range(n), 2):
                avaliable_indices = [iter([where_ones[row, 0]]) if has_ones[row] else range(k) for row in rows]
                for columns in product(*avaliable_indices):
                    i1 = rows[0]
                    i2 = rows[1]
                    j1 = columns[0]
                    j2 = columns[1]

                    _matrix = a.copy()
                    _matrix[i1, j1] = (_matrix[i1, j1] + 1) % 2
                    _matrix[i2, j2] = (_matrix[i2, j2] + 1) % 2

                    yield Ank(_matrix)

            for i in range(n):
                if has_ones[i]:
                    for j0 in range(k - 1):
                        j = j0 + (j0 >= where_ones[i])
                        _matrix = a.copy()
                        _matrix[i] = 0
                        _matrix[i, j] = 1
                        yield Ank(_matrix)


def matrix_neighborhood_generator(a: Ank, D=[2]):
    return generator(a.get_value(), a.get_n(), a.get_k(), D)
