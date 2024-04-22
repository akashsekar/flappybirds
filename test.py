import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum number of steps to run the game during testing")

    args = parser.parse_args()
    return args

def test(opt):
    torch.manual_seed(123)  # Consistent behaviour across CPU and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("{}/flappy_bird_2000000".format(opt.saved_path), map_location=device)
    model.eval()
    model.to(device)

    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    actions = []  # Initialize actions list to store actions taken
    rewards = []  # Track rewards over time
    terminals = []  # Track game terminations
    steps = 0  # Initialize step counter

    while not terminal and steps < opt.max_steps:  # Run until the game is over or the step limit is reached
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        actions.append(action)  # Log the action taken

        next_image, reward, terminal = game_state.next_frame(action)
        rewards.append(reward)  # Log the reward
        terminals.append(1 if terminal else 0)  # Log the game termination status

        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
        next_image = torch.from_numpy(next_image).to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state
        steps += 1  # Increment the step counter

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(actions, label='Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.title('Action Choices Over Time')
    plt.yticks([0, 1], ['No Flap', 'Flap'])
    plt.legend()
    plt.savefig('actions_over_time.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Rewards Received Over Time')
    plt.legend()
    plt.savefig('rewards_over_time.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(terminals, label='Game Terminations')
    plt.xlabel('Time Step')
    plt.ylabel('Game Over')
    plt.title('Game Terminations Over Time')
    plt.legend()
    plt.savefig('terminals_over_time.png')
    plt.show()

if __name__ == "__main__":
    opt = get_args()
    test(opt)
