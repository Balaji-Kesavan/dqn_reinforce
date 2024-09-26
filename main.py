def main():
    env = gym.make("Test")
    replay_buffer = ReplayBuffer(10000)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    q_network = DQN(state_size, action_size)
    target_network = DQN(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters())
    
    epsilon = 1.0  # Initial epsilon for epsilon-greedy policy
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    num_episodes = 1000
    batch_size = 64
    update_target_every = 100  # How often to update the target network
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = q_network(state_tensor).argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            
            # Train the network
            train_dqn(env, q_network, target_network, optimizer, replay_buffer, batch_size)
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update the target network
        if episode % update_target_every == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    main()
