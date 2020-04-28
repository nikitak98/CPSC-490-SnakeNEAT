from settings import *
import vision


class Snake:

    def __init__(self):
        self.color = snake_color
        self.direction = random.randint(0,3)
        self.head = (random.randint(2,width/block_size - 3)*block_size,random.randint(2,height/block_size - 3)*block_size)
        self.body = collections.deque([self.head,(self.head[0]+block_size*-direction_to_dxdy(direction)[0],self.head[1]+block_size*-direction_to_dxdy(direction)[1]),(self.head[0]+block_size*2*-direction_to_dxdy(direction)[0],self.head[1]+block_size*2*-direction_to_dxdy(direction)[1])])
        self.hunger = max_hunger
        self.len = len(self.body)

    def in_food(self,food):
        if food == self.head:
            return True
        return False

    def eat(self):
        self.hunger = max_hunger
        self.len += 1

    def in_wall(self):
        if self.head[0] < 0 or self.head[0] > width - block_size:
            return True
        elif self.head[1] < 0 or self.head[1] > height - block_size:
            return True
        return False

    def in_body(self):
        if self.body.count(self.head) > 1:
            return True
        return False

    def move_head(self,direction):
        self.head = (self.head[0] + direction_to_dxdy(direction)[0]*block_sizeself.head[1] + direction_to_dxdy(direction)[1]*block_size)
        self.body.appendleft(self.head)
        self.hunger -= 1
        self.direction = direction
        return

    def move_tail(self):
        self.body.pop()
        return

    def tail_direction(self):
        if self.body[self.len-1][0] > self.body[self.len-2][0]:
            return 1
        if self.body[self.len-1][0] < self.body[self.len-2][0]:
            return 3
        if self.body[self.len-1][1] < self.body[self.len-2][1]:
            return 0
        if self.body[self.len-1][1] > self.body[self.len-2][1]:
            return 2
