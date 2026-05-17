import torch.nn as nn
from collections import deque
import random
import torch
import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)

    def reset(self):
        self.board = np.zeros(9)
        return self.board.copy()

    def check_winner(self, player):
        b = self.board.reshape(3, 3)
        # Check rows
        for i in range(3):
            if np.all(b[i, :] == player):  # row
                return True
            if np.all(b[:, i] == player):  # column
                return True
        # Check diagonals
        if np.all(np.diag(b) == player):  # top-left to bottom-right
            return True
        if np.all(np.diag(np.fliplr(b)) == player):  # top-right to bottom-left
            return True
        return False

    def step(self, action):
        if self.board[action] != 0:  # illegal move
            return self.board.copy(), -10, True

        self.board[action] = 1
        if self.check_winner(1):
            return self.board.copy(), +1, True  # win
        if np.all(self.board != 0):
            return self.board.copy(), 0, True  # draw

        # Opponent plays randomly
        empty = np.where(self.board == 0)[0]
        opp = np.random.choice(empty)
        self.board[opp] = -1
        if self.check_winner(-1):
            return self.board.copy(), -1, True  # loss

        return self.board.copy(), 0, False


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, 9)
            # Q-value per cell
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)
        self.buffer = ReplayBuffer(50000)
        self.epsilon = 1.0
        self.gamma = 0.95

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)        # explore
        q = self.q_net(torch.FloatTensor(state))
        # Mask illegal moves
        mask = torch.full((9,), -1e9)
        mask[valid_actions] = 0
        return (q + mask).argmax().item()              # exploit

    def train_step(self, batch_size=64):
        if len(self.buffer.buffer) < batch_size:
            return
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ✅ Convert to numpy first, then to tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # Current Q
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q  (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()


# ── Improved Training ─────────────────────────────────────────────────

def get_opponent_move(board, mode="smart"):
    """Smart opponent that blocks and attacks"""
    empty = np.where(board == 0)[0].tolist()

    # Check if opponent can win → play there
    for move in empty:
        board[move] = -1
        temp_env = TicTacToe()
        temp_env.board = board.copy()
        if temp_env.check_winner(-1):
            board[move] = 0
            return move
        board[move] = 0

    # Check if AI can win → block it
    for move in empty:
        board[move] = 1
        temp_env = TicTacToe()
        temp_env.board = board.copy()
        if temp_env.check_winner(1):
            board[move] = 0
            return move
        board[move] = 0

    # Otherwise random
    return np.random.choice(empty)


def step_smart(env, action, board):
    """Custom step using smart opponent"""
    if board[action] != 0:
        return board.copy(), -10, True  # illegal

    board[action] = 1
    env.board = board.copy()
    if env.check_winner(1):
        return board.copy(), +5, True  # AI wins  ← bigger reward

    if np.all(board != 0):
        return board.copy(), +1, True  # draw is ok

    # Smart opponent
    opp_action = get_opponent_move(board)
    board[opp_action] = -1
    env.board = board.copy()

    if env.check_winner(-1):
        return board.copy(), -5, True  # AI lost  ← bigger penalty

    if np.all(board != 0):
        return board.copy(), +1, True  # draw

    return board.copy(), 0, False


# ── Training Loop ─────────────────────────────────────────────────────
agent = DQNAgent()
env = TicTacToe()
UPDATE_TARGET_EVERY = 500

for episode in range(80000):  # ← more episodes
    state = env.reset()
    done = False

    while not done:
        valid = np.where(state == 0)[0].tolist()
        action = agent.select_action(state, valid)
        next_state, reward, done = step_smart(env, action, state.copy())
        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.train_step()
        state = next_state

    # Decay epsilon more slowly
    agent.epsilon = max(0.01, agent.epsilon * 0.9999)

    if episode % UPDATE_TARGET_EVERY == 0:
        agent.target_net.load_state_dict(agent.q_net.state_dict())

    if episode % 5000 == 0:
        print(f"Episode {episode} | Epsilon: {agent.epsilon:.3f}")

torch.save(agent.q_net.state_dict(), "tictactoe_dqn.pth")
print("Model saved!")