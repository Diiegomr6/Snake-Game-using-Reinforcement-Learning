"""
Snake Eater Environment
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
import numpy as np
import random
from collections import deque


class SnakeGameEnv:
    def __init__(self, frame_size_x=150, frame_size_y=150, growing_body=True):
        # Initializes the environment with default values
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.growing_body = growing_body
        self.reset()

    def reset(self):
        # Resets the environment with default values
        self.snake_pos = [50, 50]
        self.snake_body = [[50, 50], [60, 50], [70, 50]]
        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        return self.get_state()

    def step(self, action):
        self.update_snake_position(action)

        self.game_over = self.check_game_over()

        reward = self.calculate_reward()

        self.update_food_position()
        state = self.get_state()
        return state, reward, self.game_over

     
    def get_state(self):
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos
        body = self.snake_body

        food_north = int(food_y < head_y)
        food_south = int(food_y > head_y)
        food_west = int(food_x < head_x)
        food_east = int(food_x > head_x)

        def is_danger(pos):
            x, y = pos
            if x < 0 or x >= self.frame_size_x or y < 0 or y >= self.frame_size_y:
                return True
            if [x, y] in body[1:]:
                return True
            return False

        def is_trapped(new_head):
            visited = set()
            queue = deque([new_head])
            tail = body[-1]
            temp_body = body[:-1]  # assume tail moves

            def valid(pos):
                x, y = pos
                return (0 <= x < self.frame_size_x and 0 <= y < self.frame_size_y and
                        pos not in temp_body)

            while queue:
                x, y = queue.popleft()
                if [x, y] == tail:
                    return False  # Not trapped

                for dx, dy in [(-10, 0), (10, 0), (0, -10), (0, 10)]:
                    nx, ny = x + dx, y + dy
                    next_pos = (nx, ny)
                    if next_pos not in visited and valid([nx, ny]):
                        visited.add(next_pos)
                        queue.append(next_pos)
            return True  # Trapped

        def valid_step(pos):
            return int(not is_danger(pos) and not is_trapped(pos))

        valid_step_west = valid_step([head_x - 10, head_y])
        valid_step_east = valid_step([head_x + 10, head_y])
        valid_step_north = valid_step([head_x, head_y - 10])
        valid_step_south = valid_step([head_x, head_y + 10])

        state = [
            food_north,
            food_south,
            food_west,
            food_east,
            valid_step_west,
            valid_step_east,
            valid_step_north,
            valid_step_south
        ]

        return self.encode_state(state)

    
    def encode_state(self, state_list):
        return int("".join(map(str, state_list)), 2)

    def get_body(self):
        return self.snake_body

    def get_food(self):
        return self.food_pos
    
    def calculate_reward(self):
        if self.game_over:
            return -100  # Penalty for dying
        elif self.snake_pos == self.food_pos:
            return +10  # Reward for eating food
        else:
            # Calcular distancia anterior y actual
            current_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
            previous_head = self.snake_body[1]  # Posición anterior de la cabeza
            previous_distance = abs(previous_head[0] - self.food_pos[0]) + abs(previous_head[1] - self.food_pos[1])

            if current_distance < previous_distance:
                return +1  # Se acerca
            else:
                return -1  # Se aleja
        
    def check_game_over(self):
        # Return True if the game is over, else False
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            return True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            return True
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True
                
        return False

    def update_snake_position(self, action):
        # Updates the snake's position based on the action
        # Map action to direction
        change_to = ''
        direction = self.direction
        if action == 0:
            change_to = 'UP'
        elif action == 1:
            change_to = 'DOWN'
        elif action == 2:
            change_to = 'LEFT'
        elif action == 3:
            change_to = 'RIGHT'
    
        # Move the snake
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
    
        if direction == 'UP':
            self.snake_pos[1] -= 10
        elif direction == 'DOWN':
            self.snake_pos[1] += 10
        elif direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif direction == 'RIGHT':
            self.snake_pos[0] += 10
            
        self.direction = direction
        
        
        self.snake_body.insert(0, list(self.snake_pos))
        
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 10
            self.food_spawn = False
            # If the snake is not growing
            if not self.growing_body:
                self.snake_body.pop()
        else:
            self.snake_body.pop()
    
    def update_food_position(self):
        if not self.food_spawn:
            while True:
                new_food_pos = [
                    random.randrange(1, (self.frame_size_x // 10)) * 10,
                    random.randrange(1, (self.frame_size_y // 10)) * 10
                ]
                if new_food_pos not in self.snake_body:
                    self.food_pos = new_food_pos
                    break
        self.food_spawn = True
        

