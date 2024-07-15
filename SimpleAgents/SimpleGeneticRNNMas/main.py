import pygame

from Constant import *
from GeneticEnv import Environment
from GeneticEnv import screen

def main():
    env = Environment(GRID_SIZE, NUM_AGENTS, NUM_OBSTACLES, NUM_RESOURCES)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 12)
    generation = 0

    running = True
    max_k = 720
    k = 0
    
    while generation < NUM_GENERATIONS:
        while running and k<max_k:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.update()
            env.draw(screen, font)
            pygame.display.flip()
            clock.tick(109)
            k+=1
        
        generation += 1
        env.evolve()
        clock.tick(10)
        k = 0
        
    pygame.quit()

if __name__ == "__main__":
    main()