import numpy as np
from itertools import combinations, product, chain


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


def column_range_selector(one_position, k):
    return chain(range(one_position), range(one_position + 1, k))


def brute_force_column_matrix_generator(matrix, k, where_one, rows):
    col_valid_indices = [column_range_selector(where_one[row], k) for row in rows]
    for cols in product(*col_valid_indices):
        neighbor_matrix = matrix.copy()
        neighbor_matrix[rows, cols] = 1
        yield neighbor_matrix


def append_matrix_generator(matrix, k, where_one, append_rows, distance):
    for rows in combinations(append_rows, distance):
        column_generator = brute_force_column_matrix_generator(matrix, k, where_one, rows)
        for matrix in column_generator:
            yield matrix


def _set_builder_fast(m, distances):
    """
    Task set generator (B set generator).

    This solution needs O(n*k) memory, O(n*k) precompute time and O(n*k) time per set element generation.
    O(n*k) time for generation is needed to make matrix copy. If we don't take this into account, this
    algorithm runs with O(n + k) time for every matrix.

    Args:
      matrix: 2D ndarray with shape [n, k].
      distance_set: list of distances that specifies B set.
    Generates:
      Values from B set, which contains matrices under task rules (binary n x k matrix
      with not more than one "1" in each row) with Hamming distances specified in distance_set.
    """

    n, k = np.shape(m)
    ones_count, one_pos, has_one, where_one, zero_rows = get_matrix_info(m, n)

    # run through all distances in D
    for distance in distances:

        # check that distance is correct
        if distance < 0 or distance > ones_count + n:
            continue

        for num_ones_removed in range(np.min([distance, ones_count]) + 1):
            for clear_rows in combinations(one_pos[0], num_ones_removed):

                cleared_matrix = m.copy()
                cleared_matrix[clear_rows, :] = 0
                remained_rows = np.array(list((set(range(n)) - set(one_pos[0])).union(set(clear_rows))))

                append_generator = append_matrix_generator(m, k, where_one, remained_rows,
                                                           distance - num_ones_removed)
                for matrix in append_generator:
                    yield matrix


def set_builder_fast(matrix: np.ndarray, distance_set):
    unique_distance_set = list(set(distance_set))
    return _set_builder_fast(matrix, unique_distance_set)
