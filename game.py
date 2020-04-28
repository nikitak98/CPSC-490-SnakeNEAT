import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame, random, sys
import collections
import neat
import math
import datetime
import pickle
from vision import *
import replay
from settings import *
import visualize

counter = 0
all_time_max_fitness = 0
seed_to_play = None
genome_to_play = None
play = False

dir_save = 'run' + str(datetime.datetime.now())

def eval_genomes(genomes,config):

    global counter
    global snake_color
    global all_time_max_fitness
    global seed_to_play
    global genome_to_play
    global play
    global dir_save

    for genome_id, genome in genomes:

        # Direction Variable
        # 0 = Down, 1 = Left, 2 = Up, 3 = Right
        saved_seed = random.randint(-sys.maxsize,sys.maxsize)
        random.seed(saved_seed)
        direction = random.randint(0,3)

        # Initial Parameters of Snake
        snake_head_initial = (random.randint(2,width/block_size - 3)*block_size,random.randint(2,height/block_size - 3)*block_size)
        snake_body = collections.deque([snake_head_initial,(snake_head_initial[0]+block_size*-direction_to_dxdy(direction)[0],snake_head_initial[1]+block_size*-direction_to_dxdy(direction)[1]),(snake_head_initial[0]+block_size*2*-direction_to_dxdy(direction)[0],snake_head_initial[1]+block_size*2*-direction_to_dxdy(direction)[1])])

        # Food
        food = (random.randint(0,width/block_size - 1)*block_size,random.randint(0,width/block_size - 1)*block_size)
        while snake_body.count(food) > 0:
            food = (random.randint(0,width/block_size - 1)*block_size,random.randint(0,width/block_size - 1)*block_size)

        net = neat.nn.FeedForwardNetwork.create(genome,config)
        run = True

        eaten = 0
        hunger = max_hunger
        steps = 0
        max_score = 0

        while run:

            # HEAD DIRECTION - ORDER: DOWN LEFT UP RIGHT
            # NN Inputs [4 x Head Direction, 8 x 3 x Vision (Wall Dist,Food,Body]
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
            # Clockwise starting from Down
            vision = look_direction(0,1,snake_body,food)
            input[8] = vision[0]
            input[16] = vision[1]
            input[24] = vision[2]

            vision = look_direction(-1,1,snake_body,food)
            input[9] = vision[0]
            input[17] = vision[1]
            input[25] = vision[2]

            vision = look_direction(-1,0,snake_body,food)
            input[10] = vision[0]
            input[18] = vision[1]
            input[26] = vision[2]

            vision = look_direction(-1,-1,snake_body,food)
            input[11] = vision[0]
            input[19] = vision[1]
            input[27] = vision[2]

            vision = look_direction(0,-1,snake_body,food)
            input[12] = vision[0]
            input[20] = vision[1]
            input[28] = vision[2]

            vision = look_direction(1,-1,snake_body,food)
            input[13] = vision[0]
            input[21] = vision[1]
            input[29] = vision[2]

            vision = look_direction(1,0,snake_body,food)
            input[14] = vision[0]
            input[22] = vision[1]
            input[30] = vision[2]

            vision = look_direction(1,1,snake_body,food)
            input[15] = vision[0]
            input[23] = vision[1]
            input[31] = vision[2]

            output = net.activate(input)
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
            if snake_body[0][0] < 0 or snake_body[0][0] > width - block_size/2:
                run = False
            if snake_body[0][1] < 0 or snake_body[0][1] > width - block_size/2:
                run = False

            # Check collision
            if snake_body.count(snake_body[0]) > 1:
                run = False

            # Check if in Food
            if snake_body[0] == food:
                hunger = max_hunger
                eaten += 1
                if len(snake_body) == 100:
                    print("WINNER!!!")
                    run = False
                else:
                    food = (random.randint(0,width/block_size - 1)*block_size,random.randint(0,width/block_size - 1)*block_size)
                    while snake_body.count(food) > 0:
                        food = (random.randint(0,width/block_size - 1)*block_size,random.randint(0,width/block_size - 1)*block_size)
            else: # If not spawn new food
                hunger -= 1
                snake_body.pop() # Remove last block of snake to prepare for update

            # Check if dead
            if hunger <= 0:
                run = False

            steps += 1

        genome.fitness = steps + 100 * eaten ** 2
        if genome.fitness > all_time_max_fitness:
            play = True
            all_time_max_fitness = genome.fitness
            seed_to_play = saved_seed
            genome_to_play = genome

    if play:
        folder_name = dir_save + '/generation' + str(counter)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        with open(folder_name + '/genome', 'wb') as f:
            pickle.dump(genome_to_play, f)
        with open(folder_name + '/seed', 'wb') as f:
            pickle.dump(seed_to_play,f)
        print(all_time_max_fitness)
        replay.play(genome_to_play,seed_to_play)
    play = False
    counter += 1

if __name__ == "__main__":

    pygame.init()
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

    winner = None
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    if len(sys.argv) > 1 and sys.argv[1] == '--load-checkpoint':
        p = neat.Checkpointer.restore_checkpoint(sys.argv[2])
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100,filename_prefix = dir_save + '/neat-checkpoint-'))

    winner = p.run(eval_genomes,1000)

    with open(dir_save + '/winner-snake','wb') as f:
        pickle.dump(winner,f)

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
