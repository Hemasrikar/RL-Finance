import numpy as np

class GridWorld:
    """
    A simple 4x4 GridWorld MDP.
    State: (row, col). Goal: reach (3,3). Pit: (2,2).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = (size-1, size-1)
        self.pit = (size//2, size//2)
        self.action_effects = [(-1,0),(1,0),(0,-1),(0,1)]  # U D L R

    def state_id(self, s): return s[0]*self.size + s[1]
    def state_xy(self, sid): return (sid // self.size, sid % self.size)

    def reset(self):
        self.pos = (0, 0)
        return self.state_id(self.pos)

    def step(self, action):
        dr, dc = self.action_effects[action]
        r, c = self.pos
        nr = max(0, min(self.size-1, r + dr))
        nc = max(0, min(self.size-1, c + dc))
        self.pos = (nr, nc)
        done = False
        if self.pos == self.goal:
            reward, done = +10.0, True
        elif self.pos == self.pit:
            reward, done = -10.0, True
        else:
            reward = -0.1       # small step penalty encourages efficiency
        return self.state_id(self.pos), reward, done

env = GridWorld()


def temporal_dl(env, policy, num_episodes=2000, gamma=0.99, lr=0.1):
    """
    policy: callable(state) to action
    Returns: V (value function array), td_errors (history)
    """
    V = np.zeros(env.n_states)
    td_errors_hist = []

    for _ in range(num_episodes):
        s = env.reset()
        done = False
        ep_td = []
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)

            # TD error
            td_error = r + gamma * V[s_next] * (1 - done) - V[s]

            # Update rule
            V[s] += lr * td_error
            ep_td.append(abs(td_error))
            s = s_next
        td_errors_hist.append(np.mean(ep_td))

    return V, td_errors_hist


def q_learning(env, num_episodes=5000, gamma=0.99, lr=0.1, epsilon=0.1):
    """
    Algorithm 2 (paper): Q-learning with epsilon-greedy exploration.
    Off-policy: uses max_a' Q(s',a') regardless of what action is taken.
    
    Returns: Q table, cumulative rewards per episode
    """
    Q = np.zeros((env.n_states, env.n_actions))
    cum_rewards = []

    for ep in range(num_episodes):
        s = env.reset()
        done = False
        total_r = 0
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                a = np.random.randint(env.n_actions)
            else:
                a = np.argmax(Q[s])
            
            s_next, r, done = env.step(a)
            total_r += r
            
            # Q-learning update (Eq. 2.23)
            target = r + gamma * np.max(Q[s_next]) * (1 - done)
            Q[s, a] += lr * (target - Q[s, a])
            s = s_next

        cum_rewards.append(total_r)

    return Q, cum_rewards
