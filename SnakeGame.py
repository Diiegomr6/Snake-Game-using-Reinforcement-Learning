"""
Snake Eater Game
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
from snake_env import SnakeGameEnv
from q_learning import QLearning
import pygame
import sys

def main():
    # Window size
    FRAME_SIZE_X = 450
    FRAME_SIZE_Y = 450
    
    # Colors (R, G, B)
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)
    BLUE = pygame.Color(0, 0, 255)
    

    # TRAIN 
    difficulty = 50 # Adjust as needed
    render_game = True # Show the game or not
    growing_body = True # Makes the body of the snake grow
    training = True # Defines if it should train or not

    # TEST
    # difficulty = 0 # Adjust as needed
    # render_game = True # Show the game or not
    # growing_body = False  # Makes the body of the snake grow
    # training = False # Defines if it should train or not

    # Initialize the game window, environment and q_learning algorithm
    # Your code here.
    # You must define the number of possible states.
    # number_states = whatever
    number_states = 2 ** 8  # 11 binary variables = 2^11
    num_episodes = 2000  # You can adjust this number as needed

    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    ql = QLearning(n_states=number_states, n_actions=4)
    # num_episodes = the number of episodes you want for training.


    if render_game:
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        fps_controller = pygame.time.Clock()
    
    total_rewards_sum = 0


    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False
        
        while not game_over:
            # Choose action using ε-greedy polic
            action = ql.choose_action(state,allowed_actions = [0,1,2,3], training = training) 

            # Interact with environment
            next_state, reward, game_over = env.step(action)

            if training:
                # Update Q-table
                ql.update_q_table(state, action, reward, next_state, terminal_state=game_over)

            # Update for next tick
            state = next_state
            total_reward += reward

            # Render
            if render_game:
                game_window.fill(BLACK)
                snake_body = env.get_body()
                food_pos = env.get_food()
                for pos in snake_body:
                    pygame.draw.rect(game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
                pygame.draw.rect(game_window, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

            if env.check_game_over():
                break

            if render_game:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                pygame.display.flip()
                fps_controller.tick(difficulty)

        total_rewards_sum += total_reward

        ql.save_q_table()
        print(f"Episode {episode+1}, Total reward: {total_reward}")

    average_reward = total_rewards_sum / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {average_reward:.2f}")

if __name__ == "__main__":
    main()
