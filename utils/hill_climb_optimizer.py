import random

from utils.graph_list_utils import calculate_length


def hill_climbing(tour, graph, max_iter=10, sample_size=1000):
    best_tour = tour.copy()
    best_cost = calculate_length(best_tour, graph)
    n = len(tour)

    iter_count = 0
    while iter_count < max_iter:
        improved = False
        samples = _get_random_samples(n, sample_size)
        for (i, k) in samples:
            new_tour = _swap_2opt(best_tour, i, k)
            new_cost = calculate_length(new_tour, graph)
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost
                improved = True
                break
        if not improved:
            break
        iter_count += 1

    return best_tour


def _swap_2opt(tour, i, k):
    new_tour = tour[0:i] + tour[i:k + 1][::-1] + tour[k + 1:]
    return new_tour


def _get_random_samples(n, sample_size):
    samples = []
    for _ in range(sample_size):
        i = random.randint(1, n - 2)
        k = random.randint(i + 1, n - 1)
        samples.append((i, k))
    return samples
