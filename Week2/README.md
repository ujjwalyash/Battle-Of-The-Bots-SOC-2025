# Week 2: Deep Q-Networks (DQN)

Welcome to Week 2! This week, we scale up from tabular Q-learning to Deep Q-Networks (DQNs), which use neural networks to approximate Q-values. This allows us to tackle environments with large or continuous state spaces, like CartPole and Snake.

---

## Goals

* Understand why tabular Q-learning fails in large state spaces
* Learn how DQNs approximate the Q-function using neural networks
* Implement a simple DQN in PyTorch
* Train a DQN on OpenAI Gymnasium's CartPole environment
* Extend your DQN agent to train on your own Snake game

---

## Core Concepts

### Function Approximation with Neural Networks

In DQNs, we replace the Q-table with a neural network:

$$
Q(s, a; \theta) \approx \text{expected return}
$$

Where $\theta$ are the parameters of the neural network.

### Experience Replay

Instead of learning from recent transitions only, we store experiences in a replay buffer and sample batches from it:

* Breaks correlation between subsequent steps
* Improves sample efficiency

### Target Network

We use a separate, periodically-updated target network $Q_{target}(s, a; \theta^-)$ to stabilize learning:

$$
\text{loss} = \left(Q(s, a) - \left(\text{reward} + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')\right)\right)^2
$$

### DQN Training Loop Skeleton

```python
for episode in range(num_episodes):
    state, _ = env.reset()

    for t in range(max_timesteps):
        # ε-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_net(state))

        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))

        # Sample batch and compute loss
        if len(replay_buffer) > batch_size:
            batch = sample(replay_buffer)
            optimize(q_net, target_net, batch)

        if done:
            break
        state = next_state
```

---

## Assignments

1. **Implement a DQN agent** using PyTorch on `CartPole-v1` using `gymnasium`. A starting notebook is provided.
2. **Implement a custom Snake game environment** with `gym.Env` API and train your DQN agent on it.
3. **DQN on a continuous space environment** defined as `CatMouseEnv`
4. (Bonus) **Compare performance** between your Snake and CartPole environments. Also try to compare Q learning and DQN in Snake for a reasonably big grid size (40 x 40).

---

## Recommended Hyperparameters (CartPole)

* Learning rate: 1e-3
* Batch size: 64
* Gamma: 0.99
* Replay buffer size: 10,000
* Target network update freq: every 100 steps
* ε decay: linear from 1.0 to 0.1 over 500 episodes

---

## Resources

* [RL Book (Sutton & Barto) - Chapter 6](http://incompleteideas.net/book/RLbook2020.pdf)
* [DQN Paper (Mnih et al. 2015)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Gymnasium Docs](https://gymnasium.farama.org/)
* [DQN from scratch tutorial (YouTube)](https://www.youtube.com/watch?v=wc-FxNENg9U)

---

## TLDR

Tabular methods don't scale. DQNs learn Q-values using neural networks. You'll build a DQN agent, train it on CartPole, Snake game and a custom environment, and learn how replay and target networks stabilize learning.
