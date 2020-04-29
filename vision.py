import numpy as np

from settings import *

def look_direction(direction,snake,food):

    (x_direction,y_direction) = dxdy_eight(direction)

    distance_to_wall = 0.0
    distance_to_food = np.inf
    distance_to_body = np.inf

    total_distance = 0.0
    distance = 1.0

    # Start one block out
    current_x = snake[0][0] + block_size*x_direction
    current_y = snake[0][1] + block_size*y_direction
    total_distance += distance

    food_found = False
    body_found = False

    while current_x < width and current_x >= 0 and current_y < height and current_y >= 0:
        if not food_found and (current_x,current_y) == food:
            food_found = True
            distance_to_food = total_distance
        if not body_found and (current_x,current_y) in snake:
            body_found = True
            distance_to_body = total_distance

        distance_to_wall = total_distance
        current_x += block_size * x_direction
        current_y += block_size * y_direction
        total_distance += distance

    distance_to_wall = 1.0 / total_distance
    distance_to_food = 1.0 / distance_to_food
    distance_to_body = 1.0 / distance_to_body

    return (distance_to_wall,distance_to_food,distance_to_body)

def dxdy_four(direction):

    if direction == 0:
        (dx,dy) = (0,1)
    if direction == 1:
        (dx,dy) = (-1,0)
    if direction == 2:
        (dx,dy) = (0,-1)
    if direction == 3:
        (dx,dy) = (1,0)

    return (dx,dy)

def dxdy_eight(direction):

    if direction == 0:
        (dx,dy) = (0,1)
    if direction == 1:
        (dx,dy) = (-1,1)
    if direction == 2:
        (dx,dy) = (-1,0)
    if direction == 3:
        (dx,dy) = (-1,-1)
    if direction == 4:
        (dx,dy) = (0,-1)
    if direction == 5:
        (dx,dy) = (1,-1)
    if direction == 6:
        (dx,dy) = (1,0)
    if direction == 7:
        (dx,dy) = (1,1)

    return (dx,dy)
