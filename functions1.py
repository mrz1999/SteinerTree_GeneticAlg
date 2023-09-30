from cgitb import reset
import random
import numpy
import matplotlib.pyplot as plt
import math
from deap import tools
from deap import base, creator, tools

#-------------------------------------------------Utils------------------------------------------------------#

def grafo(connections):
    temp = []
    distances = []
    for i in connections:
        temp.append([i[0],i[1]])
        #print("pairs of connected nodes = ",temp)
        #print("\n")
        distances.append(i[2])
    #print("distances = ",distances)
    # the pairs of connected points are plotted with their corresponding connection, the final mst is formed
    for pair in temp:
        plt.plot(*zip(*pair), "ro-", c = 'black')
        plt.axis(xmin=0, xmax=1, ymin=0, ymax=1)
    plt.show()
    return distances


def grafo_new(connections, steiner_point = []):
    temp = []
    distances = []
    for i in connections:
        temp.append([i[0],i[1]])
        distances.append(i[2])

    for pair in temp:
        plt.axis(xmin=0, xmax=1, ymin=0, ymax=1)
        plt.plot(*zip(*pair), "ro-", c = 'black')
    plt.axis(xmin=0, xmax=1, ymin=0, ymax=1)
    if steiner_point:    
        plt.plot(*zip(*steiner_point), 'o', c = 'red')
    plt.show()

    return distances


def n_conn(steiner_points, connections):
    number = []
    for point in steiner_points:
        count = 0
        for connection in connections: 
            if point == connection[0]:
                count += 1
            if point == connection[1]:
                count += 1
        number.append(count) 
    return number


def prim_alg(vertices):
    remaining_points = vertices.copy()                        # list of points not yet in the mst (all points at this stage)
    mst = []                                                  # list of the points in the mst (empty list at this stage)
    connections = []                                          # list of containing pairs of connected points and their respective distance
    source = random.choice(remaining_points)                  # at start a random point is selected
    remaining_points.remove(source)                           # the first point is removed from remaining_points
    mst.append(source)                                        # the first point is inserted into mst

    while remaining_points:                                   
        outer_min = 1000                                         # at each iteration,for each point in mst, the distance between itself and 
        for selected in mst:                                  # every point in remaining_points is calculated, then the point in 
            min_dist = 1000                                    # remaining_points closest to it is saved as the best next point to include;
            for possible in remaining_points:                 # we then compare the closest point found of each point in mst and the final
                d = math.dist(selected, possible)             # point chosen will be the one which distance from its corresponding point
                if d < min_dist:                              # in mst is the lowest
                    min_dist = d
                    new_point = possible
            if min_dist < outer_min:
                outer_min = min_dist
                final_b = new_point
                final_a = selected
        mst.append(final_b)                                   # the final next point is added to mst
        connections.append([final_a, final_b, outer_min])     # the final next point, its correspondent previous one and their distance are saved # in connections
        remaining_points.remove(final_b)                      # the final next point is removed from remaining_points # the process is repeated until there are not anymore elements in remaining_points
        
    return(connections)


def ind_to_points (ind):
    steiner_points = []
    i = 0
    while i < len(ind):
        steiner_points.append([ind[i]]+[ind[i+1]])
        i += 2
    return steiner_points


def evaluation(vertices, fixed_vertices):
    
    steiner_point = ind_to_points(vertices)
    allpoints = steiner_point + fixed_vertices
    connections = prim_alg(allpoints)

    total_dist = 0
    for element in connections:
        total_dist += element[2]

    num_conn = n_conn(steiner_point, connections) 
    penality = 0
    for element in num_conn:
        if element == 1 or element == 2:
            penality += 0
    total_dist += penality

    return(total_dist)


def duplicate_fixverts (fixed_vertices, POP_SIZE):
    temp = []
    for i in range(POP_SIZE):
        temp.append(fixed_vertices)
    return temp


def opt_new(steiner_point, conns):
    
    connections = conns.copy()

    counter = n_conn(steiner_point, connections)
    i = 0
    while 1 in counter or 2 in counter:
        i += 1
        #print("before while cycle {}: counter list = {}".format(i, counter))
        for index in range(len(steiner_point)):

            if counter[index] == 1: # nel caso di una connessione devo semplicemente rimuovere quell'elemento
                deleted_connection=[]
                for connection in connections:
                    if steiner_point[index] in connection:
                        deleted_connection.append(connection)

                connections.remove(deleted_connection[0])
                counter = n_conn(steiner_point, connections)
                #print("after deleting type-1 stp in cycle {}: counter list = {}".format(i, counter))
                break        


            elif counter[index] == 2: # in questo caso prima trovo le due connessioni e le metto nella lista delle cose da eliminare e poi creo la nuova connessione
                new_connection = []
                deleted_connection=[]
                new_points = []
                for connection in connections:
                    if steiner_point[index] == connection[0]: #se lo steiner point è il primo punto allora dovrò salvare il secondo per poi collegarlo
                        new_points.append(connection[1]) 
                        deleted_connection.append(connection)
                    elif steiner_point[index] == connection[1]: #viceversa di prima
                        new_points.append(connection[0])
                        deleted_connection.append(connection)

                if [new_points[0], new_points[1], math.dist(new_points[0], new_points[1])] not in connections:
                    new_connection.append([new_points[0], new_points[1], math.dist(new_points[0], new_points[1])])

                for deletion in deleted_connection:
                    connections.remove(deletion)
                connections = connections + new_connection
                counter = n_conn(steiner_point, connections)
                #print("after deleting type-2 stp in cycle {}: counter list = {}".format(i, counter))
                break
        resultantList = []
        for element in connections:
            if element not in resultantList:
                resultantList.append(element)
        connections = resultantList.copy()
    
    new_steiner = []
    for steiner,conn in zip(steiner_point, counter):
        if conn != 0:
            new_steiner.append(steiner)


    return connections, new_steiner


#--------------------------------------------Genetic Algorithm-----------------------------------------------#


def set_GA(n_steiner_point, my_sigma):
    
    # Creating Classes
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)


    IND_SIZE = 2 * (n_steiner_point)
    TOURN_SIZE = 10

    toolbox = base.Toolbox()
    toolbox.register('attr_float', lambda: random.random())

    # Individuals
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE) 

    # Population
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    ### Operators ###

    # Crossover
    toolbox.register('mate', tools.cxBlend, alpha=0.5)

    # Mutation
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=my_sigma)

    # Selection
    toolbox.register('select', tools.selTournament, tournsize=TOURN_SIZE)
    #toolbox.register('select_r', tools.selRoulette)

    # Evaluation
    toolbox.register('evaluation', evaluation)

    #Statistical Features
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def GA(toolbox, POP_SIZE, CXPB, MUTPB, NGEN, stats, fixed_vertices):
    
    #Defining Hall of Fame
    hof = tools.HallOfFame(1)
    
    #Creating the population
    pop = toolbox.population(n=POP_SIZE)
    #print(pop)

    #Defining the Logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else[])

    #Evaluate the entire population
    temp_fixed_vertices = duplicate_fixverts(fixed_vertices, POP_SIZE)
    fitness = list(map(toolbox.evaluation, pop, temp_fixed_vertices))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = [fit]


    hof.update(pop) if stats else {}

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(pop), **record)
    

    for g in range(NGEN):
        #print('Generation Number ', g, 'Population ', pop)
    
        #Select the next generation individuals
        #offspring = toolbox.select_r(pop, len(pop))
        offspring = toolbox.select(pop, len(pop))
        
        #Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        #Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2],offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant, indpb=0.5)
                del mutant.fitness.values

        #Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = list(map(toolbox.evaluation, invalid_ind, temp_fixed_vertices))
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = [fit]

        if hof is not None:
            hof.update(offspring)
            #print(hof)

        #The population in entirely replaced by the offspring
        pop[:] = tools.selBest(offspring, POP_SIZE-1)
        pop.append(hof[0])

        record = stats.compile(pop) if stats else{}
        #print(record)
        logbook.record(gen=g+1, nevals=len(invalid_ind), **record)
        

    return pop, logbook, hof


def steiner_alg(n_fixed_points, n_steiner_point, my_sigma, POP_SIZE, CXPB, MUTPB, seed = None):

    #CREATE FIXED POINTS
    fixed_vertices = []
    #fixed_vertices = [[0.1,0.1],[0.1,0.9],[0.9,0.1],[0.9,0.9]]
    #fixed_vertices = [[0.1,0.1],[0.9,0.1],[0.5,0.9]]
    #fixed_vertices = [[0.1,0.1],[0.8,0.1],[0.45,0.706]]

    if seed:    
        random.seed(seed)
    for i in range(n_fixed_points):
      
        point = [0,0]
        fixed_vertices.append(point)
        point[0] = random.random()
        point[1] = random.random()

    #-----------------------------------------------------------------------------------------------
    #GENETIC ALGORITHM

    toolbox, stats = set_GA(n_steiner_point, my_sigma)

    NGEN = 50

    GA_exe = GA(toolbox, POP_SIZE, CXPB, MUTPB, NGEN, stats, fixed_vertices)

    # best point(s) found so far
    point = GA_exe[2][0]

    #------------------------------------------------------------------------------------------------
    # EVALUATE THE STARTING DISTANCE, AND THE DISTANCE AFTER EVALUATION AND OPTIMIZATION

    connections_before = prim_alg(fixed_vertices)
    total_length_before = 0
    for connection in connections_before:
        total_length_before += connection[2]

    #DISTANCE AFTER EVALUATION
    steiner = ind_to_points(point)
    allpoints = steiner + fixed_vertices

    connections_after = prim_alg(allpoints)
    
    #OPTIMIZATION ALGORITHM
    opt_connections, opt_steiner = opt_new(steiner, connections_after)

    total_length_after = 0
    for connection in opt_connections:
        total_length_after += connection[2]

    return(total_length_before,total_length_after,connections_before,opt_connections, opt_steiner)


def test_for_steiner_number (n_fixed_points, my_sigma, POP_SIZE, CXPB, MUTPB, seed = None):
    distances = []

    for i in range(1, n_fixed_points - 1):
        #print("--- Running with {} steiner point(s) ---".format(i))
        n_steiner_point = i
        old_distance, new_distance, conn_bfr, conn_aft, opt_steiner = steiner_alg(n_fixed_points, n_steiner_point, my_sigma, POP_SIZE, CXPB, MUTPB, seed=seed)
        distances.append([old_distance, new_distance])

    for index in range(len(distances)):
        print("\n")
        print("Before distance with {} steiner point(s) = {}".format(index + 1, distances[index][0]))
        print("After distance with {} steiner point(s) = {}".format(index + 1, distances[index][1]))
        print("Difference between After and Before = ", distances[index][1] - distances[index][0])
        print("Ratio between After and Before = ", distances[index][1] / distances[index][0])
    
    distance_before = distances[0][0]
    only_distances_after = []
    for i in distances:
        only_distances_after.append(i[1])
    best_istance = min(only_distances_after)
    best_steiner_number = only_distances_after.index(best_istance) + 1
    print("\n")
    print("The best number of steiner points was {} with:\n".format(best_steiner_number))
    print("Before distance = ", distance_before)
    print("After distance = ", best_istance)
    print("Difference between After and Before = ", distance_before - best_istance)
    print("Ratio between After and Before = ", best_istance / distance_before)