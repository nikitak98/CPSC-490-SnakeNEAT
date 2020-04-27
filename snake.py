from settings import *
import vision


class Snake:

    def __init__(self):
        self.color = snake_color
        self.direction = random.randint(0,3)
        self.head = (random.randint(2,width/block_size - 3)*block_size,random.randint(2,height/block_size - 3)*block_size)
        self.body = collections.deque([self.head,(self.head[0]+block_size*-direction_to_dxdy(direction)[0],self.head[1]+block_size*-direction_to_dxdy(direction)[1]),(self.head[0]+block_size*2*-direction_to_dxdy(direction)[0],self.head[1]+block_size*2*-direction_to_dxdy(direction)[1])])
        self.hunger = max_hunger
