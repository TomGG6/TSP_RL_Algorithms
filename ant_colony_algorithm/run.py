import time

from ant_colony_algorithm.ant_colony import AntColony
from ant_colony_algorithm.graph import Graph


def run_ant_colony_algorithm(file_path):
    graph = Graph(file_path)

    # Parametry algorytmu
    num_ants = 20
    num_iterations = 100
    num_elite = 2
    rho = 0.1
    alpha = 1
    beta = 5


    start_time = time.time()
    colony = AntColony(num_ants, num_iterations, num_elite, rho, alpha, beta, graph)
    colony.run()
    stop_time = time.time()

    exec_time = stop_time - start_time
    # Wyświetlenie najlepszej znalezionej ścieżki
    # print("Najlepsza odległość:", colony.best_distance)
    # print("Najlepsza ścieżka:", colony.best_path)
    return colony.best_distance, exec_time

if __name__ == "__main__":
    print(run_ant_colony_algorithm('../data/att48.txt'))
