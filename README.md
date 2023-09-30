# SteinerTree_GeneticAlg
We have developed a Genetic Algorithm for solving the Steiner Tree problem.  Briefly, we want to solve the problem of finding the shortest path among a series of points with the possibility of adding further point that, if strategically positioned, can be used for obtaining a path shorter that the one obtained considering just the starting points.
For further explication about the Steiner Tree problem you can see the file steiner_tree.pdf.In the file is also presented the developed algorithm with the chosen parameter for each part of the algorithm. (NOTE that in the implementation of the genetic algorithm are used the heuristic of the Steiner problem described in the pdf).

Prim's Algorithm is used for calculating the shortest path among the series of starting points and it is used as threshold for calculating the improvement of the genetic algorithm.
A complete implementation of the Prim's Algorithm is presented in the file primalg.ipynb.

In the file functions1.py are implemented the utils needed for the implementation of the genetic algorithm for the Steiner tree problem.

For the implementation of the genetic algorithm is used the library deap. For the documentation of deap: https://deap.readthedocs.io/en/master/
The genetic algorithm is defined in the functions set_GA; the body of the genetic algorithm is in the function GA. Both functions are implemented in the functions1.py file.

For playing with parameters and implementation of the Genetic Algorithm you can use the file prototype.ipynb.
More intuitively, you can use the full_alg.py file and testing the algorithm with different combinations of parameters.



