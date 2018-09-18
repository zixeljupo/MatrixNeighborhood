import numpy as np
from itertools import combinations, product, chain
import math
import numbers


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
    return num_values*4//(1024*1024)


# Calculates max fast calculation depth for matrix [n x k] with maximum memory usage ram_limit MB.
# Fast calculation depth is a value that is used fast part of the algorithm.
# It specifies how many "one" insertions we could do to store all insertion combinations,
# and obtained matrices in memory.
def calculate_max_depth(n, k, ram_limit=2):

    depth = 1
    complexity = k
    matrix_size = n*k

    while values_to_mb(complexity*(matrix_size + depth)) <= ram_limit:
        depth += 1
        complexity *= k

    return depth - 1


# Fast part of the algorithm.
# Creates cache with matrices, each belongs to B.
# Matrices are created by "one" insertions in rows specified in variable "rows".
# Cache size depends on how many rows there are exponentially, approximately like k^len(rows)
# If there already was "one" in some row before, this place is ignored for "one" insertion.
def create_matrix_cache_with_one_additions(m, n, k, rows, one_position):

    # generate column indices for passed rows
    column_indices = [get_zero_indices(one_position[row], k) for row in rows]

    rows = np.array(rows, dtype=np.int32)
    rows_count = rows.size

    # generate product of column_indices, obtained shape is [prod(len(column_indices[i])), rows_count]
    indices = np.array(np.meshgrid(*column_indices)).T.reshape(-1, rows_count)
    cache_size = np.shape(indices)[0]

    # adding offsets to indices that shifts rows and matrices in cache
    indices += rows * k
    indices = (indices.T + np.arange(cache_size) * (n * k)).T
    indices = indices.reshape(indices.size)

    # creating and filling cache
    matrix_cache = np.tile(m, (cache_size, 1, 1))
    matrix_cache.put(indices, 1)

    #print("Mb for cache: {}".format(values_to_mb(matrix_cache.size + indices.size)))

    return matrix_cache


def set_builder_core(m, distances, ram_limit=2):
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

    # lazy computation for max depth
    fast_max_depth = None

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

        for num_ones_removed in range(np.min([distance, ones_count]) + 1):
            for one_remove_rows in combinations(one_pos[0], num_ones_removed):

                m1 = m.copy()
                m1[one_remove_rows, :] = 0
                r1 = list(set(zero_rows).union(set(one_remove_rows)))

                new_distance = distance - num_ones_removed
                fast_depth = np.min([fast_max_depth, new_distance])
                slow_depth = new_distance - fast_depth

                for rows in combinations(r1, new_distance):

                    slow_rows = rows[:slow_depth]
                    slow_col_indices = [chain(range(where_one[i]), range(where_one[i] + 1, k)) for i in slow_rows]
                    for slow_cols in product(*slow_col_indices):

                        m2 = m1.copy()
                        m2.put(np.array(slow_rows, dtype=np.int32)*k + np.array(slow_cols, dtype=np.int32), 1)
                        r2 = np.array(list(set(rows) - set(slow_rows)))

                        for fast_rows in combinations(r2, fast_depth):

                            if len(fast_rows) == 0:
                                yield m2.copy()
                                continue

                            matrix_cache = create_matrix_cache_with_one_additions(m2, n, k, fast_rows, where_one)
                            for m3 in matrix_cache:
                                yield m3


def set_builder(matrix: np.ndarray, distances, ram_limit=2):

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
