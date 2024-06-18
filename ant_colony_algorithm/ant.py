import numpy as np


class Ant:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.tour = []  # Lista odwiedzonych miast
        self.visited = np.zeros(num_cities, dtype=bool)  # Tablica wizytowanych miast
        self.tour_length = 0.0  # Długość trasy mrówki

    def visit_city(self, city_idx, distance):
        self.tour.append(city_idx)
        self.visited[city_idx] = True
        self.tour_length += distance

    def reset(self):
        self.tour = []
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.tour_length = 0.0