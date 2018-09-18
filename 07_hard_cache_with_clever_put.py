import numpy as np
from itertools import combinations, product, chain
import math


# checks that passed value is matrix
def is_matrix(value):
    return type(value) == np.ndarray and len(np.shape(value)) == 2


# checks that matrix contains only zeros and ones
def is_matrix_binary(matrix):
    return matrix.min() >= 0 and matrix.max() <= 1


# checks that matrix has correct structure
def is_matrix_correct(matrix):
    return max(matrix.sum(axis=1)) <= 1


def get_matrix_info(matrix, n):
    """
    Matrix info generator.

    Takes O(n*k) time.
    Calculates some values for given matrix before B set generation.

    Args:
      matrix: 2D ndarray with shape [n, k].
      n: number of matrix rows.
    Returns:
      Some variables that are used by B set generator.
    """

    ones_positions = np.asarray(np.where(matrix == 1))
    ones_count = np.shape(ones_positions)[1]

    one_position_in_row = np.full(n, -1, dtype=int)
    np.put(one_position_in_row, ones_positions[0], ones_positions[1])
    has_one_in_row = np.select([one_position_in_row != -1], [1]).astype(bool)
    zero_rows = np.asarray(np.where(np.invert(has_one_in_row))).squeeze(0)

    return ones_count, ones_positions, has_one_in_row, one_position_in_row, zero_rows


def get_zero_indices(one_pos, k):
    return list(range(one_pos)) + list(range(one_pos + 1, k))


# this is the fastest solution I've found
def _set_builder_fast(m, distances):
    """
    Task set generator (B set generator)..

    Args:
      matrix: 2D ndarray with shape [n, k].
      distance_set: list of distances that specifies B set.
    Generates:
      Values from B set, which contains matrices under task rules (binary n x k matrix
      with not more than one "1" in each row) with Hamming distances specified in distance_set.
    """

    if not is_matrix(m):
        raise ValueError("Expected numpy matrix")

    if not is_matrix_binary(m):
        raise ValueError("Expected binary matrix")

    if not is_matrix_correct(m):
        raise ValueError("Expected matrix that has valid structure (not more than one '1' in row)")

    n, k = np.shape(m)
    ones_count, one_pos, has_one, where_one, zero_rows = get_matrix_info(m, n)

    # run through all distances in D
    for distance in distances:

        # check that distance is correct
        if distance < 0 or distance > ones_count + n:
            continue

        if distance == 0:
            yield m.copy()
            continue

        for num_ones_removed in range(np.min([distance, ones_count]) + 1):
            for clear_rows in combinations(one_pos[0], num_ones_removed):

                cleared_matrix = m.copy()
                cleared_matrix[clear_rows, :] = 0
                remained_rows = np.array(list((set(range(n)) - set(one_pos[0])).union(set(clear_rows))))

                for rows in combinations(remained_rows, distance - num_ones_removed):

                    if len(rows) == 0:
                        yield cleared_matrix.copy()
                        continue

                    avaliable_column_indices = [get_zero_indices(where_one[row], k) for row in rows]

                    rows = np.array(rows, dtype=np.int32)
                    rows_count = rows.size

                    if (k ** rows_count)*n*k*4//1024//1024 > 1024:
                        raise ValueError("Looks that you don't have enough RAM")

                    indices = np.array(np.meshgrid(*avaliable_column_indices)).T.reshape(-1, rows_count)

                    cache_size = np.shape(indices)[0]

                    indices += rows*k
                    indices = (indices.T + np.arange(cache_size)*(n*k)).T
                    indices = indices.reshape(indices.size)

                    matrix_cache = np.tile(cleared_matrix, (cache_size, 1, 1))
                    matrix_cache.put(indices, 1)

                    # print("Mb for cache: " + str((cache_size*rows_count + cache_size*n*k)*4//1024//1024))

                    for mat in matrix_cache:
                        yield mat


def set_builder_fast(matrix: np.ndarray, distance_set):
    unique_distance_set = list(set(distance_set))
    return _set_builder_fast(matrix, unique_distance_set)
