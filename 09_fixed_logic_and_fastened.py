import numpy as np
from itertools import combinations, product, chain
from scipy.special import comb
import math
import numbers


def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        int, count=count * k)
    return index.reshape(-1, k)


# check that passed value is matrix
def is_matrix(value):
    return type(value) == np.ndarray and len(np.shape(value)) == 2


# check that matrix contains only zeros and ones
def is_matrix_binary(matrix):
    return matrix.min() >= 0 and matrix.max() <= 1


# check that matrix has correct structure
def is_matrix_correct(matrix):
    return max(matrix.sum(axis=1)) <= 1


# check that passed value is integer-like
def is_integral(value):
    return isinstance(value, numbers.Integral)


# converts distance set values to list
# removes non-unique values
# checks that set doesn't have non-integral values
def set_to_distance_list(distance_set):
    unique_distance_list = list(set(distance_set))

    for distance in unique_distance_list:
        if not is_integral(distance):
            raise ValueError("Found bad distance type in distance_set: {}".format(distance))

    return unique_distance_list


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


# returns list of int that has values from 0 to k-1
# if one_pos >= 0, it is excluded from the list
def get_zero_indices(one_pos, k):
    return list(range(one_pos)) + list(range(one_pos + 1, k))


# returns amount of ram in MB that is needed to store num_values 4-byte values
def values_to_mb(num_values):
    return num_values * 4 // (1024 * 1024)


# Calculates max fast calculation depth for matrix [n x k] with maximum memory usage ram_limit MB.
# Fast calculation depth is a value that is used fast part of the algorithm.
# It specifies how many "one" insertions we could do to store all insertion combinations,
# and obtained matrices in memory.
def calculate_max_depth(n, k, ram_limit=2):
    depth = 1
    complexity = k
    matrix_size = n * k

    while values_to_mb(complexity * (matrix_size + depth)) <= ram_limit:
        depth += 1
        complexity *= k

    return depth - 1


def set_builder_core(m, distances, ram_limit=0):
    """
    B set generator.

    Args:
      m: source matrix, stored in 2D ndarray with shape [n, k].
      distances: list of distances that specifies B set.
      ram_limit: maximal amount of ram that program can use (in MB)
    Generates:
      Values from B set, which contains matrices under task rules (binary n x k matrix
      with not more than one "1" in each row) with Hamming distances specified in distance_set.
    """

    n, k = np.shape(m)
    ones_count, one_pos, has_one, where_one, zero_rows = get_matrix_info(m, n)
    matrix_size = n * k

    # lazy computation for max depth
    fast_max_depth = None
    zero_indices = None

    # run through all distances in D
    for distance in distances:

        if distance < 0 or distance > ones_count + n:
            continue
        if distance == 0:
            yield m.copy()
            continue
        if distance == 1:
            for i in range(n):
                if has_one[i]:
                    m1 = m.copy()
                    m1[i] = 0
                    yield m1
                else:
                    for j in range(k):
                        m1 = m.copy()
                        m1[i, j] = 1
                        yield m1
            continue

        # not needed before
        if fast_max_depth is None:
            fast_max_depth = calculate_max_depth(n, k, ram_limit=ram_limit)
            zero_indices = np.array([np.array(get_zero_indices(where_one[row], k), dtype=np.int32) for row in range(n)])

        for num_ones_removed in range(np.min([distance, ones_count]) + 1):
            for one_remove_rows in combinations(one_pos[0], num_ones_removed):

                m1 = m.copy()
                m1[one_remove_rows, :] = 0
                new_zero_rows = np.append(zero_rows, np.array(one_remove_rows, dtype=np.int32))
                new_distance = distance - num_ones_removed
                if new_distance == 0:
                    yield m1
                    continue

                fast_depth = np.min([fast_max_depth, new_distance])
                slow_depth = new_distance - fast_depth

                for one_append_rows in combinations(new_zero_rows, new_distance):

                    slow_rows = one_append_rows[:slow_depth]
                    fast_rows = one_append_rows[slow_depth:]

                    slow_rows_np = np.array(slow_rows, dtype=np.int32) * k
                    fast_rows_np = np.array(fast_rows, dtype=np.int32) * k

                    slow_col_indices = [zero_indices[row] for row in slow_rows]
                    fast_col_indices = [zero_indices[row] for row in fast_rows]

                    cache_size = int(np.prod(np.array([col.size for col in fast_col_indices], dtype=np.int32)))

                    for slow_cols in product(*slow_col_indices):
                        m2 = m1.copy()
                        m2.put(slow_rows_np + np.array(slow_cols, dtype=np.int32), 1)
                        matrix_cache = np.tile(m2, (cache_size, 1, 1))
                        matrix_cache.put((np.array(np.meshgrid(*fast_col_indices, copy=False)).T.reshape(-1, fast_depth)
                                          + fast_rows_np).T + (np.arange(cache_size) * matrix_size), 1)
                        for m3 in matrix_cache:
                            yield m3


def set_builder(matrix: np.ndarray, distances, ram_limit=0):
    if not is_matrix(matrix):
        raise ValueError("Expected numpy ndarray with two dimensions, got\n{}".format(matrix))

    if not is_matrix_binary(matrix):
        raise ValueError("Expected binary matrix, got\n{}".format(matrix))

    if not is_matrix_correct(matrix):
        raise ValueError("Expected matrix that has valid structure (not more than one '1' in row)")

    if ram_limit < 0:
        raise ValueError("Ram can't be negative: {}".format(ram_limit))

    distance_list = set_to_distance_list(distances)

    return set_builder_core(matrix, distance_list, ram_limit)
