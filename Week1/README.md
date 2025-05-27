# Week 1: Q-Learning Fundamentals

Welcome to Week 1! This week, we dive into Q-Learning — the foundational algorithm of value-based reinforcement learning. There's an [ipynb notebook](./Week1.ipynb) to get you started.

---

## Goals

* Understand what Q-learning is and how it fits in the RL landscape
* Learn the Bellman Equation and how it's used to update Q-values
* Implement tabular Q-learning from scratch in Python
* Explore exploration vs. exploitation via ε-greedy strategy
* Train a Q-learning agent on OpenAI Gym’s FrozenLake or Taxi-v3 environment

---

## Core Concepts

### Markov Decision Process (MDP)

An environment for RL is often modeled as an MDP with:

* **States (S)**
* **Actions (A)**
* **Transition probabilities (T)**
* **Rewards (R)**
* **Discount factor (γ)**

### Q-Function (Action-Value Function)

The Q-function estimates the *expected future rewards* of taking an action `a` in a state `s`:

$$
Q(s, a) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a\right]
$$

### Bellman Equation (Q-Learning Update Rule)

The Q-value is updated as:

```python
Q[s, a] = Q[s, a] + α * (reward + γ * max(Q[next_state]) - Q[s, a])
```

Where:

* α: learning rate
* γ: discount factor
* max(Q\[next\_state]): estimated value of the best action from the next state

### Exploration vs Exploitation

Use ε-greedy to balance exploring new actions and exploiting known good ones:

```python
if random.random() < epsilon:
    action = random.choice(action_space)
else:
    action = argmax(Q[state])
```

---

## Training Loop Skeleton (Tabular Q-learning)

```python
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
```

---

## Challenges

1. **Implement tabular Q-learning** on multi-Arm Bandits, boilerplate code given in the ipynb
2. **Try different ε decay strategies** and observe convergence speed.
3. Train an agent on **Taxi-v3**, which has a larger state space. Track average reward per episode.
4. (Bonus) Visualize the learned Q-table after training and interpret the agent's behavior.

---

## Resources    

* [RL Book (Sutton & Barto) - upto Chapter 5:](http://incompleteideas.net/book/RLbook2020.pdf)
* [David Silver RL Course - Lecture 4 (Model-Free Prediction)](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
* [OpenAI Gym Docs](https://www.gymlibrary.dev/)
* [RL: Q-learning explained clearly (YouTube)](https://www.youtube.com/watch?v=0g4j2k_Ggc4)

---

## TLDR

Understand the Q-table, Bellman update rule, and how ε-greedy works. Your goal is to master the **Q-learning loop**, visualize convergence, and debug policy errors.