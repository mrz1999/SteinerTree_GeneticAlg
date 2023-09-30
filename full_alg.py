from functions1 import *

def main():

    n_fixed_points = 10
    n_steiner_point = 8
    my_sigma = 0.3
    POP_SIZE = 400
    CXPB = 0.9
    MUTPB = 0.1
    seed = None
    
    test_for_steiner_number(n_fixed_points, my_sigma, POP_SIZE, CXPB, MUTPB, seed=seed)

    # before_distance, after_distance, before_connections, after_connections, steiner = steiner_alg(n_fixed_points, n_steiner_point, my_sigma, POP_SIZE, CXPB, MUTPB, seed=None)
    # print("\n")
    # print("MST length = ", before_distance)
    # print("MEStT length = ", after_distance)
    # print("Difference = ", before_distance - after_distance)
    # print("Ratio = ", after_distance/before_distance)
    # print("\n")
    
    # print("Minumum Spanning Tree of fixed points")
    # grafo_new(before_connections)

    # print("Optimized Steiner Tree of fixed points")
    # grafo_new(after_connections, steiner)


if __name__ == '__main__':
  main()