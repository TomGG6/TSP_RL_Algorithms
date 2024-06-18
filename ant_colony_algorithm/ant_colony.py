import numpy as np

from ant_colony_algorithm.ant import Ant


class AntColony:
    def __init__(self, num_ants, num_iterations, num_elite, rho, alpha, beta, graph):
        self.num_ants = num_ants  # Liczba mrówek
        self.num_iterations = num_iterations  # Liczba iteracji
        self.num_elite = num_elite  # Liczba elitarnych mrówek
        self.rho = rho  # Parametr odparowania feromonu
        self.alpha = alpha  # Waga feromonu
        self.beta = beta  # Waga odległości
        self.graph = graph  # Graf

        self.best_path = []  # Najlepsza znaleziona ścieżka
        self.best_distance = float('inf')  # Najmniejsza znaleziona odległość

        self.pheromone = np.ones((graph.num_cities, graph.num_cities))  # Tablica feromonu

    def run(self):
        for _ in range(self.num_iterations):
            ants = [Ant(self.graph.num_cities) for _ in range(self.num_ants)]
            for ant in ants:
                self._run_ant(ant)
                if ant.tour_length < self.best_distance:
                    self.best_distance = ant.tour_length
                    self.best_path = ant.tour.copy()
            self._update_pheromone(ants)
            self._pheromone_evaporation()

    def _run_ant(self, ant):
        start_city = np.random.randint(self.graph.num_cities)
        ant.visit_city(start_city, 0)
        current_city = start_city
        for _ in range(self.graph.num_cities - 1):
            next_city = self._select_next_city(ant, current_city)
            distance = self.graph.distances[current_city][next_city]
            ant.visit_city(next_city, distance)
            current_city = next_city
        distance_back = self.graph.distances[current_city][start_city]
        ant.visit_city(start_city, distance_back)

    def _select_next_city(self, ant, current_city):
        pheromone = self.pheromone[current_city] ** self.alpha
        visibility = 1 / (self.graph.distances[current_city] + 1e-10) ** self.beta
        probabilities = pheromone * visibility
        probabilities[ant.visited] = 0
        probabilities /= np.sum(probabilities)
        next_city = np.random.choice(range(self.graph.num_cities), p=probabilities)
        return next_city

    def _update_pheromone(self, ants):
        pheromone_delta = np.zeros((self.graph.num_cities, self.graph.num_cities))
        for ant in ants:
            for i in range(self.graph.num_cities - 1):
                city1, city2 = ant.tour[i], ant.tour[i + 1]
                pheromone_delta[city1][city2] += 1 / ant.tour_length
                pheromone_delta[city2][city1] += 1 / ant.tour_length
        self.pheromone = (1 - self.rho) * self.pheromone + pheromone_delta

    def _pheromone_evaporation(self):
        self.pheromone *= self.rho