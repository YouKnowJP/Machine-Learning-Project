# main.py

import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
import config


def main():
    agent = Agent()
    env = Environment()
    plot_rewards = []
    num_episodes = config.NUM_EPISODES

    env.start_game()  # Countdown before starting; switch to the game window

    try:
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            state = env.reset()
            total_reward = 0
            step_counter = 0
            done = False
            episode_start_time = time.time()

            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)

                # Store experience more aggressively after early game steps
                if step_counter > 700:
                    for _ in range(5):
                        agent.remember(state, next_state, action, reward, done, step_counter)
                elif step_counter > 40:
                    agent.remember(state, next_state, action, reward, done, step_counter)
                if done:
                    # Extra experiences when game ends
                    for _ in range(10):
                        agent.remember(state, next_state, action, reward, done, step_counter)
                    print("Game over")
                    break

                state = next_state
                step_counter += 1
                total_reward += reward

            if step_counter > 0:
                avg_fps = step_counter / (time.time() - episode_start_time)
                print("Avg Frame-Rate:", avg_fps)
            plot_rewards.append(total_reward)
            print("Episode Reward:", total_reward)

            agent.learn()

            if episode % 20 == 0:
                torch.save(agent.model.state_dict(), config.PRETRAINED_WEIGHTS)
                print("Saved model weights to", config.PRETRAINED_WEIGHTS)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Plot episode rewards
    plt.figure()
    plt.plot(range(len(plot_rewards)), plot_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    # Plot training loss
    plt.figure()
    plt.plot(range(len(agent.loss_history)), agent.loss_history)
    plt.title("Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
