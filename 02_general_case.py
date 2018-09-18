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


def _set_builder_slow(matrix, distance_set):
    """
    Task set generator (B set generator).

    This solution needs O(n*k) memory, O(n*k) precompute time and O(n*k) time per set element generation.
    O(n*k) time for generation is needed to make matrix copy. If we don't take this into account, this
    algorithm runs with O(n + k) time.

    Args:
      matrix: 2D ndarray with shape [n, k].
      distance_set: list of distances that specifies B set.
    Generates:
      Values from B set, which contains matrices under task rules (binary n x k matrix
      with not more than one "1" in each row) with Hamming distances specified in distance_set.
    """

    n, k = np.shape(matrix)
    ones_count, one_pos, has_one, where_one, zero_rows = get_matrix_info(matrix, n)

    # run through all distances in D
    for distance in distance_set:

        # check that distance is correct
        if distance < 0 or distance > ones_count + n:
            continue

        # number of rows where Hamming distance between source matrix and result matrix is 2
        num_twice_modified = np.min([distance//2, ones_count])
        while num_twice_modified != -1:

            # number of rows where Hamming distance between source matrix and result matrix is 1
            num_once_modified = distance - num_twice_modified*2

            # if we don't have enough rows to perform modifications
            if num_once_modified + num_twice_modified > n:
                break

            for twice_modified_rows in combinations(one_pos[0], num_twice_modified):

                binary_cols_idx = [iter(list(set(range(k)) - {where_one[row]})) for row in twice_modified_rows]
                remained_ones_rows = np.asarray(list(set(one_pos[0]) - set(twice_modified_rows)))

                for twice_modified_cols in product(*binary_cols_idx):

                    for once_modified_rows in combinations(chain(zero_rows, remained_ones_rows), num_once_modified):

                        unary_cols_idx = [iter([where_one[row]]) if has_one[row] else range(k)
                                          for row in once_modified_rows]

                        for once_modified_cols in product(*unary_cols_idx):

                            # this is the next matrix that belongs to B
                            neighbor_matrix = matrix.copy()

                            for index in range(len(twice_modified_rows)):
                                i = twice_modified_rows[index]
                                j = twice_modified_cols[index]
                                neighbor_matrix[i] = 0
                                neighbor_matrix[i, j] = (neighbor_matrix[i, j] + 1) % 2

                            for index in range(len(once_modified_rows)):
                                i = once_modified_rows[index]
                                j = once_modified_cols[index]
                                neighbor_matrix[i, j] = (neighbor_matrix[i, j] + 1) % 2

                            yield neighbor_matrix

            num_twice_modified -= 1


def set_builder_slow(matrix: np.ndarray, distance_set):
    unique_distance_set = list(set(distance_set))
    return _set_builder_slow(matrix, unique_distance_set)
