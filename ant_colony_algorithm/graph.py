import numpy as np


class Graph:
    def __init__(self, file_path):
        self.coords = self.load_coordinates(file_path)
        self.num_cities = len(self.coords)
        self.distances = self.calculate_distances(self.coords)

    def load_coordinates(self, file_path):
        coords = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                coords.append(list(map(float, data[1:])))
        return coords

    def calculate_distances(self, coords):
        num_cities = len(coords)
        distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distances[i][j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
        return distances