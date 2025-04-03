from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import EduMentorEnv
import os

def main():
    # Create the custom environment instance
    env = EduMentorEnv(grid_size=5)

    # Initialize the DQN model with fine-tuned hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
       learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1
    )

    # Train the model
    total_timesteps = 500000  # Increased training timesteps for better learning
    print("Starting DQN training...")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    model.learn(total_timesteps=total_timesteps)
    print("Training completed!")

    # Save the trained model
    save_path = os.path.join("models", "dqn", "dqn_model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved at {save_path}")

    # Evaluate the trained model
    print("Evaluating the trained model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()