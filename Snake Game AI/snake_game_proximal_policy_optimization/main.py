# main.py

from ppo_agent import PPOAgent
import config
import os

def train_ppo():
    agent = PPOAgent()
    num_updates = config.NUM_UPDATES
    for update in range(num_updates):
        agent.train(rollout_steps=config.ROLLOUT_STEPS)
        print(f"PPO update {update+1}/{num_updates} completed.")
        # Save model periodically
        if (update + 1) % config.SAVE_INTERVAL == 0:
            if not os.path.exists(config.MODEL_FOLDER):
                os.makedirs(config.MODEL_FOLDER)
            model_path = f"{config.MODEL_FOLDER}/{config.MODEL_FILE}"
            agent.model.save(model_path)

if __name__ == '__main__':
    train_ppo()
