import random
import math
import time


def get_city(file_path):
    cities = []
    with open(file_path) as f:
        for i in f.readlines():
            node_city_val = i.split()
            cities.append([node_city_val[0], float(node_city_val[1]), float(node_city_val[2])])
    return cities


def distance(city1, city2):
    return math.sqrt((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2)


def total_distance(cities):
    return sum(distance(cities[i], cities[i + 1]) for i in range(len(cities) - 1)) + distance(cities[0], cities[-1])


def initial_population(cities, population_size):
    return [random.sample(cities, len(cities)) for _ in range(population_size)]


def tournament_selection(population, k=5):
    best = None
    for i in range(k):
        indiv = random.choice(population)
        if best is None or total_distance(indiv) < total_distance(best):
            best = indiv
    return best


def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    fill = [item for item in parent2 if item not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = fill.pop(0)
    return child


def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]


def genetic_algorithm(file_path, population_size=100, generations=500, mutation_rate=0.01):
    cities = get_city(file_path)
    population = initial_population(cities, population_size)

    for _ in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = ordered_crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    best_route = min(population, key=total_distance)
    return total_distance(best_route)


def run_genetic_algorithm(file_path):
    start_time = time.time()
    cost = genetic_algorithm(file_path)
    stop_time = time.time()
    exec_time = stop_time - start_time
    return cost, exec_time
