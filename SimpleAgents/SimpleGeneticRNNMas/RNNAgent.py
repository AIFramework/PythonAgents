import random
import numpy as np

class RNNAgent:
    def __init__(self, grid_size, vision_range=40, hidden_size=40, genes=None):
        self.grid_size = grid_size
        self.position = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]
        self.resources_collected = 0
        self.vision_range = vision_range
        self.hidden_size = hidden_size
        self.genes = genes if genes is not None else self.random_genes()
        self.hidden_state = np.zeros(hidden_size)
        self.set_weights()

    def random_genes(self):
        num_genes = 3 * self.hidden_size + self.hidden_size * self.hidden_size + self.hidden_size + self.hidden_size * 4 + 4
        return 2 * np.random.rand(num_genes+1) - 1

    def set_weights(self):
        self.w_input = self.genes[:3 * self.hidden_size].reshape(self.hidden_size, 3)
        self.w_hidden = self.genes[3 * self.hidden_size:3 * self.hidden_size + self.hidden_size ** 2].reshape(self.hidden_size, self.hidden_size)
        self.b_hidden = self.genes[3 * self.hidden_size + self.hidden_size ** 2:3 * self.hidden_size + self.hidden_size ** 2 + self.hidden_size]
        self.w_output = self.genes[3 * self.hidden_size + self.hidden_size ** 2 + self.hidden_size:3 * self.hidden_size + self.hidden_size ** 2 + self.hidden_size + self.hidden_size * 4].reshape(4, self.hidden_size)
        self.b_output = self.genes[-5:-1]

    def move(self, resources, obstacles):
        move_probabilities = self.calculate_move_probabilities(resources)
        move = self.choose_move(move_probabilities, obstacles)
        self.position = move

    def calculate_move_probabilities(self, resources):
        nearby_resources = [
            r for r in resources if self.dist(r) <= self.vision_range*np.abs(self.genes[-1])
        ]
        
        if nearby_resources:
            cm_x = sum(r[0]*(1.0/(self.dist(r)+0.01))**2 for r in nearby_resources) / len(nearby_resources)
            cm_y = sum(r[1]*(1.0/(self.dist(r)+0.01))**2 for r in nearby_resources) / len(nearby_resources)
        else:
            cm_x, cm_y = self.position

        vector_to_cm = [cm_x - self.position[0], cm_y - self.position[1]]
        input_vector = np.array([vector_to_cm[1], vector_to_cm[1] * vector_to_cm[0], vector_to_cm[0]])

        # Обновление контекста (Скрытого состояния RNN)
        self.hidden_state = np.tanh(np.dot(self.w_input, input_vector) + np.dot(self.w_hidden, self.hidden_state) + self.b_hidden)

        # Рассчет вероятностей движений на выходе
        output_vector = np.dot(self.w_output, self.hidden_state) + self.b_output
        move_probabilities = self.softmax(output_vector)

        return move_probabilities

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)
        
    def dist(self, obj):
        return np.linalg.norm(np.array(obj) - np.array(self.position))
        
    def choose_move(self, move_probabilities, obstacles):
        moves = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        for _ in range(10):
            move_idx = np.random.choice(4, p=move_probabilities)
            move = moves[move_idx]
            new_position = [self.position[0] + move[0], self.position[1] + move[1]]
            if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
                if tuple(new_position) not in obstacles:
                    return new_position
        return self.position

    def collect_resource(self):
        self.resources_collected += 1