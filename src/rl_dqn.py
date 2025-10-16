from __future__ import annotations
import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995, batch_size: int = 64, memory_size: int = 5000):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = self._build(lr)

    def _build(self, lr):
        m = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.n_actions, activation='linear')
        ])
        m.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return m

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.model.predict(state[None, :], verbose=0)[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s = np.array([b[0] for b in batch])
        a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch])
        s2 = np.array([b[3] if b[3] is not None else np.zeros_like(batch[0][0]) for b in batch])
        done = np.array([b[4] for b in batch]).astype(np.float32)

        target = self.model.predict(s, verbose=0)
        target_next = self.model.predict(s2, verbose=0)
        max_next = np.max(target_next, axis=1)
        for i in range(self.batch_size):
            target[i, a[i]] = r[i] + (1 - done[i]) * self.gamma * max_next[i]
        self.model.fit(s, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
