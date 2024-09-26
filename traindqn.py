def train_dqn(env, q_network, target_network, optimizer, replay_buffer, batch_size, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return
    
    # Sample a batch of experiences from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Convert them to PyTorch tensors
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)
    
    # Compute Q-values for current states
    q_values = q_network(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values for next states using the target network
    with torch.no_grad():
        next_q_values = target_network(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_value = rewards + gamma * max_next_q_values * (1 - dones)
    
    # Calculate loss between predicted and target Q-values
    loss = nn.MSELoss()(q_value, target_q_value)
    
    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
