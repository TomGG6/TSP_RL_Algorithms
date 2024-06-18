import numpy as np


def convert_to_list(file):
    graph_instance = []
    with open(file, "r") as file:
        for line in file:
            v = line.split()
            graph_instance.append([float(v[1]), float(v[2])])
    return graph_instance


def calculate_length(tour, graph):
    total_length = 0.0
    for i in range(len(tour) - 1):
        point1 = graph[tour[i]]
        point2 = graph[tour[i + 1]]
        total_length += euclidean_distance(point1, point2)
    return total_length


def euclidean_distance(point1, point2):
    return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
