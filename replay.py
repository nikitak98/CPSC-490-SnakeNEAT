import os
import sys
import random
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import collections
import neat
import pickle

from vision import *
from settings import *

def play(genome,s = None):

    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

    winner_net = neat.nn.FeedForwardNetwork.create(genome, config)

    screen = pygame.display.set_mode((width,height))

    # Initial Parameters of Snake
    # Direction Variable
    # 0 = Down, 1 = Left, 2 = Up, 3 = Right
    if not s:
        saved_seed = random.randint(-sys.maxsize,sys.maxsize)
    else:
        saved_seed = int(s)
    random.seed(saved_seed)
    direction = random.randint(0,3)

    # Snake
    direction = random.randint(0,3)
    snake_head_initial = (random.randint(2,width/block_size - 3)*block_size,\
                        random.randint(2,height/block_size - 3)*block_size)
    snake_body = collections.deque([snake_head_initial])
    snake_body.append((snake_head_initial[0] + block_size * -dxdy_four(direction)[0],\
                    snake_head_initial[1] + block_size * -dxdy_four(direction)[1]))
    snake_body.append((snake_head_initial[0] + block_size * 2 * -dxdy_four(direction)[0],\
                    snake_head_initial[1] + block_size * 2 * -dxdy_four(direction)[1]))

    # Food
    food = (random.randint(0,width/block_size - 1)*block_size,\
            random.randint(0,height/block_size - 1)*block_size)
    while snake_body.count(food) > 0:
        food = (random.randint(0,width/block_size - 1)*block_size,\
                random.randint(0,height/block_size - 1)*block_size)

    # Draw
    pygame.draw.rect(screen,food_color,(food[0],food[1],block_size-1,block_size-1))
    pygame.draw.rect(screen,snake_head_color,(snake_body[0][0],snake_body[0][1],block_size-1,block_size-1))
    for (x,y) in snake_body:
        if (x,y) != snake_body[0]:
            pygame.draw.rect(screen,snake_color,(x,y,block_size-1,block_size-1))
    pygame.display.update()

    run = True
    eaten = 0
    hunger = max_hunger
    steps = 0

    pygame.time.set_timer(pygame.USEREVENT, tick_rate)
    clock = pygame.time.Clock()

    while run:

        event = pygame.event.wait()
        if event.type == pygame.USEREVENT:

            # NN Inputs [4 x Head Direction, 8 x 3 x Vision (Wall Dist,Food,Body]

            # HEAD DIRECTION
            input = 32 * [0]
            input[direction] = 1

            # TAIL DIRECTION
            body_len = len(snake_body)
            (tail_x,tail_y) = snake_body[body_len-1]
            (tail2_x,tail2_y) = snake_body[body_len-2]
            if tail_x == tail2_x: #UP/DOWN
                if tail_y > tail2_y: # UP
                    input[6] = 1
                else: #DOWN
                    input[4] = 1
            else: #LEFT/RIGHT
                if tail_x > tail2_x: # LEFT
                    input[5] = 1
                else: # RIGHT
                    input[7] = 1

            # VISION
            for i in range(0,8):
                vision = look_direction(i,snake_body,food)
                for j in range(0,3):
                    input[8 + i + j*8] = vision[j]

            output = winner_net.activate(input)
            direction = output.index(max([i for i in output]))

            # Update body
            if direction == 0:
                snake_body.appendleft((snake_body[0][0],snake_body[0][1] + block_size))
            elif direction == 1:
                snake_body.appendleft((snake_body[0][0] - block_size,snake_body[0][1]))
            elif direction == 2:
                snake_body.appendleft((snake_body[0][0],snake_body[0][1] - block_size))
            elif direction == 3:
                snake_body.appendleft((snake_body[0][0] + block_size,snake_body[0][1]))

            # Check out of bounds
            if snake_body[0][0] < 0 or snake_body[0][0] >= width:
                run = False
            if snake_body[0][1] < 0 or snake_body[0][1] >= height:
                run = False

            # Check collision
            if snake_body.count(snake_body[0]) > 1:
                run = False

            # Spawn Food
            if snake_body[0] == food:
                eaten += 1
                hunger = max_hunger
                if len(snake_body) == world_size:
                    run = False
                else:
                    food = (random.randint(0,width/block_size - 1)*block_size,\
                            random.randint(0,height/block_size - 1)*block_size)
                    while snake_body.count(food) > 0:
                        food = (random.randint(0,width/block_size - 1)*block_size,\
                                random.randint(0,height/block_size - 1)*block_size)
            else:
                hunger -= 1
                snake_body.pop()

            # Check if dead
            if hunger <= 0:
                run = False

            # Draw
            screen.fill(background_color)
            pygame.draw.rect(screen,food_color,(food[0],food[1],block_size-1,block_size-1))
            pygame.draw.rect(screen,snake_head_color,(snake_body[0][0],snake_body[0][1],block_size-1,block_size-1))
            for (x,y) in snake_body:
                if (x,y) != snake_body[0]:
                    pygame.draw.rect(screen,snake_color,(x,y,block_size-1,block_size-1))
            pygame.display.update()

            steps += 1

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: replay.py genome_file [seed_file]")
        exit()

    winner = None
    load_seed = None
    with open(sys.argv[1], 'rb') as f:
        winner = pickle.load(f)
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'rb') as f:
            load_seed = pickle.load(f)
    play(winner,load_seed)
