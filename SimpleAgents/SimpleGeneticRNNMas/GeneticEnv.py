import pygame
import numpy as np
import random

from Constant import *
from RNNAgent import RNNAgent


pygame.init()

# Размер окна
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Агенты и Ресурсы")

# Среда
class Environment:
    def __init__(self, grid_size, num_agents, num_obstacles, num_resources):
        self.grid_size = grid_size
        self.agents = [RNNAgent(grid_size) for _ in range(num_agents)]
        self.obstacles = self.generate_obstacles(num_obstacles)
        self.resources = self.generate_resources(num_resources)
        self.num_resources = num_resources

    def generate_obstacles(self, num_obstacles):
        obstacles = set()
        while len(obstacles) < num_obstacles:
            obstacle = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if all(obstacle != tuple(agent.position) for agent in self.agents):
                obstacles.add(obstacle)
        return list(obstacles)

    def generate_resources(self, num_resources):
        resources = set()
        while len(resources) < num_resources:
            resource = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if all(resource != tuple(agent.position) for agent in self.agents) and resource not in self.obstacles:
                resources.add(resource)
        return list(resources)

    def update(self):
        for agent in self.agents:
            agent.move(self.resources, self.obstacles)
            if tuple(agent.position) in self.resources:
                agent.collect_resource()
                self.resources.remove(tuple(agent.position))

    def draw(self, screen, font):
        screen.fill(BACKGROUND_COLOR)
        # Нарисовать препятствия
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, (obstacle[0] * CELL_SIZE, obstacle[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        # Нарисовать ресурсы
        for resource in self.resources:
            pygame.draw.rect(screen, RESOURCE_COLOR, (resource[0] * CELL_SIZE, resource[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        # Нарисовать агентов и количество собранных ресурсов
        for agent in self.agents:
            pygame.draw.rect(screen, AGENT_COLOR, (agent.position[0] * CELL_SIZE, agent.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            text_surface = font.render(str(agent.resources_collected), True, TEXT_COLOR)
            screen.blit(text_surface, (agent.position[0] * CELL_SIZE + 10, agent.position[1] * CELL_SIZE + 10))

    def evolve(self):
        # Сортировка агентов по количеству собранных ресурсов
        self.agents.sort(key=lambda agent: agent.resources_collected, reverse=True)
        
        # Выбор топ агентов
        num_survivors = NUM_AGENTS // 4
        survivors = self.agents[:num_survivors]
        
        # Скрещевание и мутация
        new_agents = []
        for i in range(NUM_AGENTS):
            parent1, parent2 = random.sample(survivors, 2)
            child_genes = self.crossover(parent1.genes, parent2.genes)
            child_genes = self.mutate(child_genes)
            new_agent = RNNAgent(self.grid_size, genes=child_genes)
            new_agents.append(new_agent)
        
        self.agents = new_agents
        self.resources = self.generate_resources(NUM_RESOURCES)

    def crossover(self, genes1, genes2):
        '''Скрещевание'''
        crossover_point = random.randint(0, len(genes1) - 1)
        child_genes = np.concatenate((genes1[:crossover_point], genes2[crossover_point:]))
        return child_genes

    def mutate(self, genes):
        '''Мутация'''
        for i in range(len(genes)):
            if random.random() < MUTATION_RATE:
                genes[i] += 2*random.random()-1
        return genes