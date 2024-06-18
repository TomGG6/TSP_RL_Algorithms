import time

from utils.graph_list_utils import calculate_length, convert_to_list
from utils.lkh_optimizer import lkh_optimizer


def run_lkh_algorithm(graph):
    graph = convert_to_list(graph)
    start_time = time.time()
    tour = [i for i in range(len(graph))]
    tour.append(tour[0])
    best_tour = lkh_optimizer(tour, graph, "LKH", runs=10)
    stop_time = time.time()
    exec_time = stop_time - start_time
    return calculate_length(best_tour, graph), exec_time
