from time import time
from pstats import Stats
import pstats
import cProfile


def error_test():
    bad_args_list = [[np.array([[1]], dtype=np.int32), {1}, -2],
                     [np.array([[1]], dtype=np.int32), {1}, 0],
                     [np.array([1], dtype=np.int32), {1}, 0],
                     [[[1]], {1}, 0],
                     [np.array([[2]], dtype=np.int32), {1}, 0],
                     [np.array([[-1]], dtype=np.int32), {1}, 0],
                     [np.array([[1, 1]], dtype=np.int32), {1}, 0],
                     [np.array([[1, 0], [1, 0]], dtype=np.int32), {1}, 0],
                     [np.array([[1]], dtype=np.int32), {-1, 4}, 1024],
                     [np.array([[1]], dtype=np.int32), {-1, "k"}, 1024],
                     [np.array([[1]], dtype=np.int32), {-1, range(0)}, 1024],
                     [np.array([[1]], dtype=np.int32), {1}, "k"]]
    for args in bad_args_list:
        try:
            _ = set_builder(args[0], args[1], args[2])
        except Exception as e:
            print("args: " + str(args) + ", error: " + str(e))


def current_milli_time():
    return int(round(time() * 1000))


def test_speed():

    source_matrix = np.array(
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
        dtype=int
    )
    # source_matrix = np.array(
    #     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #     dtype=int
    # )
    # source_matrix = np.array(
    #     [[0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0]],
    #     dtype=int
    # )
    #D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #source_matrix = np.zeros([10, 10], dtype=np.int32)
    D = [10]

    t = current_milli_time()
    B = set_builder(source_matrix, D, ram_limit=0)
    i = 0
    for _ in B:
        #pass
        #print(_)
        i += 1
        #assert (source_matrix != _).astype(int).sum() in D

    total_ms = current_milli_time() - t
    print("{}ms ({}s) for {} matrices.".format(total_ms, total_ms//1000, i))
    #print("{}ms ({}s) total.".format(total_ms, total_ms // 1000))


def profile_speed():

    profile = cProfile.Profile()
    try:
        profile.enable()
        test_speed()
        profile.disable()
    finally:
        stats = pstats.Stats(profile)
        stats.sort_stats("cumtime").print_stats(20)


# example usage
if __name__ == "__main__":

    test_speed()
    error_test()

    source_matrix = np.array(
        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],
        dtype=int
    )
    # source_matrix = np.array(
    #     [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    #     dtype=int
    # )
    D = [3]

    print("a:\n{source_matrix}".format(source_matrix=source_matrix))

    B = set_builder(source_matrix, D)

    print("\nB:")
    set_power = 0
    for b in B:

        set_power += 1

        print(b)
        print()

        assert (source_matrix != b).astype(int).sum() in D

    print("|B| = {set_power}.".format(set_power=set_power))
