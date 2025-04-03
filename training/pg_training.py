import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from training.environment.custom_env import EduMentorEnv

def main():
    # Create the custom environment instance
    env = EduMentorEnv(grid_size=5)  # Ensure grid size matches your environment

    # Initialize the PPO model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,  # Optimized learning rate
        n_steps=1024,  # Number of steps to run for each environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs to optimize the surrogate loss
        gamma=0.98,  # Discount factor for future rewards
        verbose=1  # Enable verbose logging
    )

    # Train the model
    total_timesteps = 300000  # Total training timesteps
    print("Starting PPO training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed!")

    # Save the trained model
    save_path = os.path.join("models", "pg", "ppo_model")
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