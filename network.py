import numpy as np
import copy
import math


def activation(arr):
    return 1 / (1 + math.exp(-arr))
    # if arr > 0:
    #     return 1
    # return 0


MUTATION_PROBABILITY = 0.2
MUTATION_WIDTH = 0.2
MUTATION_RATE = 0.5
CROSS_OVER_PROBABILITY = 0.3

activation_vectorize = np.vectorize(activation)


class Network:
    def __init__(self, input_width, layers_number, layer_width, weights=None):
        self.input_width = input_width
        self.layers_number = layers_number
        self.width = layer_width

        self.weights = []
        self.biases = []
        dimensions = [input_width] + layer_width
        if weights is None:
            for i in range(0, layers_number):
                self.weights.append(np.random.uniform(-1, 1, [dimensions[1 + i], dimensions[0 + i]]))
                self.biases.append(np.random.uniform(-1, 1, [dimensions[1 + i]]))
        else:
            index = 0
            for i in range(layers_number):
                step = dimensions[1 + i] * dimensions[0 + i]
                self.weights.append(
                    np.reshape(weights[index:index + step], [dimensions[1 + i], dimensions[0 + i]]))
                index += step
                step = dimensions[1 + i]
                self.biases.append(weights[index:index + step])
                index += step

    def evaluate(self, input):
        tmp = input
        for i in range(0, self.layers_number):
            tmp = np.sum(self.weights[i] * tmp, axis=1) + self.biases[i]
            tmp = activation_vectorize(tmp)
        return tmp[0]

    def get_weights(self):
        length = 0
        for i in range(self.layers_number):
            length += self.weights[i].shape[0] * self.weights[i].shape[1]
            length += self.biases[i].shape[0]
        layers_flatten = np.zeros(length)
        index = 0
        for i in range(0, self.layers_number):
            step = self.weights[i].shape[0] * self.weights[i].shape[1]
            layers_flatten[index:index + step] = np.ndarray.flatten(self.weights[i])
            index += step
            step = self.biases[i].shape[0]
            layers_flatten[index:index + step] = self.biases[i]
            index += step

        return layers_flatten

    def copy(self):
        return copy.deepcopy(self)


class Network_manager:
    def __init__(self, network_number, input_width, layer_number, layer_width):
        self.network_number = network_number
        self.input_width = input_width
        self.layer_number = layer_number
        self.layer_width = layer_width

        self.networks = [Network(self.input_width, self.layer_number, self.layer_width) for _ in
                         range(self.network_number)]
        self.max_fitness = 0

    def evaluate(self, input):
        steering = np.zeros(self.network_number)
        for i in range(self.network_number):
            if input[i, 0] == 1:
                steering[i] = self.networks[i].evaluate(input[i, 1:])
        return steering

    def new_generation(self, fitness):
        fitness_sum = np.sum(fitness)
        new_generation = []

        for i in range(int(self.network_number / 2)):
            # roulette selection of pair of networks
            roulette_point1 = fitness_sum * np.random.rand()
            roulette_point2 = fitness_sum * np.random.rand()
            net_index1 = 0
            net_index2 = 0
            net_sum1 = fitness[0]
            net_sum2 = fitness[0]
            while net_sum1 < roulette_point1:
                net_index1 += 1
                net_sum1 += fitness[net_index1]
            while net_sum2 < roulette_point2:
                net_index2 += 1
                net_sum2 += fitness[net_index2]

            weights1 = self.networks[net_index1].get_weights()
            weights2 = self.networks[net_index2].get_weights()

            # crossing-over
            if np.random.rand() < CROSS_OVER_PROBABILITY:
                crossing_point = np.random.randint(len(weights1))
                swap = np.zeros(crossing_point)
                swap = weights1[:crossing_point]
                weights1[:crossing_point] = weights2[:crossing_point]
                weights2[:crossing_point] = swap

            # mutation
            if np.random.rand() < MUTATION_PROBABILITY:
                indexes = np.random.randint(len(weights1), size=int(len(weights1) * MUTATION_WIDTH))
                for ind in indexes:
                    weights1[ind] += np.random.uniform(-MUTATION_RATE, MUTATION_RATE)
            if np.random.rand() < MUTATION_PROBABILITY:
                indexes = np.random.randint(len(weights1), size=int(len(weights2) * MUTATION_WIDTH))
                for ind in indexes:
                    weights2[ind] += np.random.uniform(-MUTATION_RATE, MUTATION_RATE)
            new_generation.append(Network(self.input_width, self.layer_number, self.layer_width, weights1))
            new_generation.append(Network(self.input_width, self.layer_number, self.layer_width, weights2))

        # copy best network
        new_generation[0] = self.networks[np.argmax(fitness)]
        self.networks = new_generation
