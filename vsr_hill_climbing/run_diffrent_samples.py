import numpy as np
import time

from utils.graph_list_utils import calculate_length, euclidean_distance, convert_to_list
from utils.hill_climb_optimizer import hill_climbing


def choose_initial_tour(graph):
    tour = []
    for i in range(len(graph)):
        tour.append(i)
    tour.append(0)
    return tour


def create_candidate_set(graph, n=5):
    candidate_set = {}
    for i, point in enumerate(graph):
        distances = [(j, euclidean_distance(point, other_point)) for j, other_point in enumerate(graph) if i != j]
        closest_points = sorted(distances, key=lambda x: x[1])[:n]
        candidate_set[i] = [index for index, _ in closest_points]
    return candidate_set


def epsilon_greedy(Q, candidate_set, current_point, epsilon):
    if np.random.rand() < epsilon:
        next_point = np.random.choice(candidate_set[current_point])
    else:
        q_values = {a: Q.get((current_point, a), -np.inf) for a in candidate_set[current_point]}
        max_q_value = max(q_values.values())
        if max_q_value == -np.inf:
            next_point = np.random.choice(candidate_set[current_point])
        else:
            max_q_actions = [action for action, q in q_values.items() if q == max_q_value]
            next_point = np.random.choice(max_q_actions)
    return next_point


def update_Q_value(Q, s, a, r, s_next, a_next, candidate_set, method, alpha=0.1, gamma=0.95):
    current_Q = Q.get((s, a), 0)

    if s_next in candidate_set and candidate_set[s_next]:
        if method == 1:
            # Q-learning
            next_max_Q = max(Q.get((s_next, next_a), 0) for next_a in candidate_set[s_next])
        elif method == 2:
            # SARSA
            next_max_Q = Q.get((s_next, a_next), 0)
        elif method == 4:
            if s_next not in Q:
                Q[s_next] = {action: 0 for action in candidate_set[s_next]}
            next_Q = Q.get((s_next, a_next), 0)
            next_max_Q = next_Q
        elif method == 5:
            next_max_Q = sum(Q.get((s_next, next_a), 0) for next_a in candidate_set[s_next]) / len(candidate_set[s_next])
        else:
            next_max_Q = 0
    else:
        next_max_Q = 0

    if method in [1, 2, 4, 5]:
        Q[(s, a)] = current_Q + alpha * (r + gamma * next_max_Q - current_Q)
    else:
        Q[(s, a)] = current_Q + alpha * (r - current_Q)


def update_Q_value_expected(Q, s, a, r, s_next, candidate_set, method, alpha=0.1, gamma=0.95):
    current_Q = Q.get((s, a), 0)

    if s_next in candidate_set and candidate_set[s_next]:
        if method == 1:
            # Q-learning
            next_max_Q = max(Q.get((s_next, next_a), 0) for next_a in candidate_set[s_next])
        elif method == 2:
            # Expected-SARSA
            next_max_Q = sum(Q.get((s_next, next_a), 0) for next_a in candidate_set[s_next]) / len(candidate_set[s_next])
        else:
            # MonteCarlo
            next_max_Q = 0
    else:
        next_max_Q = 0

    if method in [1, 2, 4, 5]:
        Q[(s, a)] = current_Q + alpha * (r + gamma * next_max_Q - current_Q)
    else:
        Q[(s, a)] = current_Q + alpha * (r - current_Q)


def run_vsr_hill_climbing_algorithm(graph, algorithm, epsilon=0.4, beta=0.99, alpha=0.1, gamma=0.9, M=1, num_M=3, is_vsr=True, optimum=None, sample_size=1000):
    graph = convert_to_list(graph)
    start_time = time.time()
    max_trials = len(graph)
    max_num = int(max_trials / 20)
    best_tour = choose_initial_tour(graph)
    candidate_set = create_candidate_set(graph, n=5)
    Q = {}
    num = 0

    for trial in range(max_trials):
        epsilon *= beta

        if is_vsr and num >= max_num:
            M = M % num_M + 1
            num = 0
        num += 1
        better_tour = best_tour.copy()

        for i in range(len(better_tour) - 1):
            current_point = better_tour[i]
            next_point = epsilon_greedy(Q, candidate_set, current_point, epsilon)
            reward = -euclidean_distance(graph[current_point], graph[next_point])

            a_next = epsilon_greedy(Q, candidate_set, next_point, epsilon)
            update_Q_value(Q, current_point, next_point, reward, next_point, a_next, candidate_set, M, alpha, gamma)

        better_tour = hill_climbing(better_tour, graph, sample_size=sample_size)

        if calculate_length(better_tour, graph) < calculate_length(best_tour, graph):
            best_tour = better_tour
            num = 0

        if optimum is not None and calculate_length(best_tour, graph) <= optimum:
            break
    stop_time = time.time()
    exec_time = stop_time - start_time
    return calculate_length(best_tour, graph), exec_time


def run_vsr_expected_hill_climbing_algorithm(graph, algorithm, epsilon=0.4, beta=0.99, alpha=0.1, gamma=0.9, M=1, num_M=3, is_vsr=True, optimum=None, sample_size=1000):
    graph = convert_to_list(graph)
    start_time = time.time()
    max_trials = len(graph)
    max_num = int(max_trials / 20)
    best_tour = choose_initial_tour(graph)
    candidate_set = create_candidate_set(graph, n=5)
    Q = {}
    num = 0

    for trial in range(max_trials):
        epsilon *= beta

        if is_vsr and num >= max_num:
            M = M % num_M + 1
            num = 0
        num += 1
        better_tour = best_tour.copy()

        for i in range(len(better_tour) - 1):
            current_point = better_tour[i]
            next_point = epsilon_greedy(Q, candidate_set, current_point, epsilon)
            reward = -euclidean_distance(graph[current_point], graph[next_point])
            update_Q_value_expected(Q, current_point, next_point, reward, next_point, candidate_set, M, alpha, gamma)

        better_tour = hill_climbing(better_tour, graph, sample_size=sample_size)

        if calculate_length(better_tour, graph) < calculate_length(best_tour, graph):
            best_tour = better_tour
            num = 0

        if optimum is not None and calculate_length(best_tour, graph) <= optimum:
            break
    stop_time = time.time()
    exec_time = stop_time - start_time
    return calculate_length(best_tour, graph), exec_time
