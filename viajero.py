from random import random
from random import shuffle
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt


# (x,y) Son las coordenadas de la ciudad
class Ciudad:

    def __init__(self):
        # dimensión horizontal de un punto
        self.x = 1000 * random()
        # dimensión vertical de un punto
        self.y = 1000 * random()

    def __repr__(self):
        return '(%s,%s)' % (round(self.x, 1), round(self.y, 1))


class RecorridoSimple:

    def __init__(self):
        self.tour = []

    def set_tour(self, tour):
        self.tour.extend(tour)

    def swap(self, index1, index2):
        self.tour[index1], self.tour[index2] = self.tour[index2], self.tour[index1]

    # esta es la función de costo
     # Cuanto menor sea este valor, mejor será la configuración.
    def get_distance(self):
        tour_distance = 0

        for i in range(len(self.tour)):
            tour_distance += self.distance(self.tour[i % len(self.tour)],
                                           self.tour[(i + 1) % len(self.tour)])

        return tour_distance

    @staticmethod
    def distance(city1, city2):

        dist_x = abs(city1.x - city2.x)
        dist_y = abs(city1.y - city2.y)

        return np.sqrt(dist_x * dist_x + dist_y * dist_y)

    def generate_tour(self, n):
        for _ in range(n):
            self.tour.append(Ciudad())

        shuffle(self.tour)

    def get_tour_size(self):
        return len(self.tour)

    def __repr__(self):
        return ''.join(str(e) for e in self.tour)


class RecocidoSimulado:

     # si la velocidad de enfriamiento es grande: consideramos solo unos pocos estados en el espacio de búsqueda
     # la velocidad de enfriamiento controla la cantidad de estados que considerará el algoritmo
    def __init__(self, num_cities, min_temp, max_temp, cooling_rate=0.0001):
        self.num_cities = num_cities
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.cooling_rate = cooling_rate
        self.actual_state = RecorridoSimple()
        self.next_state = None
        self.best_state = None

    def run(self):
        self.actual_state.generate_tour(self.num_cities)
        print('Distancia inicial (aleatoria): %s m' % round(self.actual_state.get_distance(), 3))

        self.best_state = self.actual_state
        temp = self.max_temp

        while temp > self.min_temp:
           # genera el estado vecino
            new_state = self.generate_random_state(self.actual_state)

           #calcula las energías (distancias)
            actual_energy = self.actual_state.get_distance()
            new_energy = new_state.get_distance()

            if random() < self.accept_prob(actual_energy, new_energy, temp):
                single_tour = RecorridoSimple()
                single_tour.set_tour(new_state.tour)
                self.actual_state = single_tour

            if self.actual_state.get_distance() < self.best_state.get_distance():
                single_tour = RecorridoSimple()
                single_tour.set_tour(self.actual_state.tour)
                self.best_state = single_tour

            temp *= 1 - self.cooling_rate

        print('Solución encontrada: %s m' % round(self.best_state.get_distance(), 3))

    @staticmethod
    def generate_random_state(actual_state):
        new_state = RecorridoSimple()
        new_state.set_tour(actual_state.tour)
        #tenemos que intercambiar 2 ciudades al azar
        random_index1 = randint(0, new_state.get_tour_size())
        random_index2 = randint(0, new_state.get_tour_size())

        new_state.swap(random_index1, random_index2)

        return new_state

    def plot_solution(self):
        xs = []
        ys = []

        self.best_state.tour.append(self.best_state.tour[0])

        for city in self.best_state.tour:
            xs.append(city.x)
            ys.append(city.y)

        # estas son las ciudades (puntos)
        plt.scatter(xs, ys)
        # conectamos las ciudades (y trazamos el ciclo hamiltoniano más corto)
        plt.plot(xs, ys)
        plt.show()

    @staticmethod
    def accept_prob(actual_energy, next_energy, temp):

        if next_energy < actual_energy:
            return 1

        return np.exp((actual_energy - next_energy) / temp)

if __name__ == '__main__':

    algorithm = RecocidoSimulado(30, 1e-9, 100000)#(num_ciudades, min_temp, max_temp)
    algorithm.run()
    algorithm.plot_solution()