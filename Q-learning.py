import random
import pickle

WINS = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6]
]


class TicTacToe:

    def __init__(self):
        self.board = [" "] * 9
        self.done = False
        self.winner = None

    def reset(self):
        self.board = [" "] * 9
        self.done = False
        self.winner = None

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == " "]

    def make_move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            return True
        return False

    def check_winner(self, player):
        for combo in WINS:
            if all(self.board[i] == player for i in combo):
                self.done = True
                self.winner = player
                return True
        if " " not in self.board:
            self.done = True
        return False

    def is_winner(self, board, player):
        for combo in WINS:
            if all(board[i] == player for i in combo):
                return True
        return False


def encode_board(board):
    mapping = {"X": 1, "O": -1, " ": 0}
    return tuple(mapping[c] for c in board)


class QLearningAgent:

    def __init__(self):
        self.q_table = {}
        self.alpha = 0.3       # learning rate — higher = learns faster
        self.gamma = 0.95      # discount — high so future rewards matter
        self.epsilon = 1.0     # start fully random

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update(self, state, action, reward, next_state, next_moves):
        future_q = (
            max(self.get_q(next_state, a) for a in next_moves)
            if next_moves else 0.0
        )
        old_q = self.get_q(state, action)
        self.q_table[(state, action)] = old_q + self.alpha * (
            reward + self.gamma * future_q - old_q
        )

    def choose_action(self, state, moves):
        if random.random() < self.epsilon:
            return random.choice(moves)
        q_values = [self.get_q(state, a) for a in moves]
        max_q = max(q_values)
        best = [moves[i] for i in range(len(moves)) if q_values[i] == max_q]
        return random.choice(best)


env = TicTacToe()
agent = QLearningAgent()


def smart_opponent_move():
    """Plays a strong opponent: win > block > center > corner > random."""

    moves = env.available_moves()

    # WIN
    for move in moves:
        env.board[move] = "O"
        if env.is_winner(env.board, "O"):
            env.board[move] = " "
            return move
        env.board[move] = " "

    # BLOCK
    for move in moves:
        env.board[move] = "X"
        if env.is_winner(env.board, "X"):
            env.board[move] = " "
            return move
        env.board[move] = " "

    # CENTER
    if 4 in moves:
        return 4

    # CORNERS
    corners = [m for m in [0, 2, 6, 8] if m in moves]
    if corners:
        return random.choice(corners)

    return random.choice(moves)


EPISODES = 600000  # more episodes = better learning

for episode in range(EPISODES):

    env.reset()
    state = encode_board(env.board)

    while not env.done:

        moves = env.available_moves()

        # --- AI (X) MOVE ---
        action = agent.choose_action(state, moves)
        env.make_move(action, "X")

        if env.check_winner("X"):
            # WIN: terminal state, no future
            agent.update(state, action, reward=1.0, next_state=None, next_moves=[])
            break

        if not env.available_moves():
            # DRAW after AI move
            agent.update(state, action, reward=0.3, next_state=None, next_moves=[])
            break

        # --- OPPONENT (O) MOVE ---
        opp_move = smart_opponent_move()
        env.make_move(opp_move, "O")

        if env.check_winner("O"):
            # LOSS: punish hard, no future
            agent.update(state, action, reward=-1.0, next_state=None, next_moves=[])
            break

        if not env.available_moves():
            # DRAW after opponent move
            agent.update(state, action, reward=0.3, next_state=None, next_moves=[])
            break

        # --- CONTINUE ---
        next_state = encode_board(env.board)
        next_moves = env.available_moves()

        # small positive reward for staying alive
        agent.update(state, action, reward=0.0, next_state=next_state, next_moves=next_moves)

        state = next_state

    # EPSILON DECAY — faster decay so AI exploits sooner
    agent.epsilon = max(0.01, agent.epsilon * 0.999985)

    if episode % 50000 == 0:
        print(f"Episode {episode:>7} | epsilon: {agent.epsilon:.4f} | q-table size: {len(agent.q_table)}")


with open("qtable.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("\nTraining complete!")
print(f"Q-table size: {len(agent.q_table)}")

import numpy as np
import random
import pickle

# ── Environment ───────────────────────────────────────────────────────
class TicTacToe:
    def __init__(self):
        self.board = tuple([0] * 9)  # tuple so it can be used as dict key

    def reset(self):
        self.board = tuple([0] * 9)
        return self.board

    def get_valid(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def make_move(self, action, player):
        b = list(self.board)
        b[action] = player
        self.board = tuple(b)

    def check_winner(self, player):
        b = self.board
        wins = [
            (0,1,2), (3,4,5), (6,7,8),  # rows
            (0,3,6), (1,4,7), (2,5,8),  # cols
            (0,4,8), (2,4,6)             # diagonals
        ]
        return any(b[i]==b[j]==b[k]==player for i,j,k in wins)

    def is_full(self):
        return 0 not in self.board

    def print_board(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        b = self.board
        print()
        print(f" {symbols[b[0]]} | {symbols[b[1]]} | {symbols[b[2]]} ")
        print("---+---+---")
        print(f" {symbols[b[3]]} | {symbols[b[4]]} | {symbols[b[5]]} ")
        print("---+---+---")
        print(f" {symbols[b[6]]} | {symbols[b[7]]} | {symbols[b[8]]} ")
        print()


# ── Q-Table Agent ─────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=1.0):
        self.Q = {}           # Q-table: {(state, action): value}
        self.alpha = alpha    # learning rate
        self.gamma = gamma    # discount factor
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def get_best_action(self, state, valid_actions):
        q_vals = {a: self.get_q(state, a) for a in valid_actions}
        max_q = max(q_vals.values())
        # if multiple actions have same Q, pick randomly among them
        best = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best)

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)   # explore
        return self.get_best_action(state, valid_actions)  # exploit

    def update(self, state, action, reward, next_state, done):
        old_q = self.get_q(state, action)

        if done:
            target = reward
        else:
            next_valid = [i for i, x in enumerate(next_state) if x == 0]
            if next_valid:
                next_max = max(self.get_q(next_state, a) for a in next_valid)
            else:
                next_max = 0.0
            target = reward + self.gamma * next_max

        # Bellman update
        self.Q[(state, action)] = old_q + self.alpha * (target - old_q)

    def save(self, path="qtable.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)
        print(f"✅ Q-table saved! ({len(self.Q)} entries)")

    def load(self, path="qtable.pkl"):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
        print(f"✅ Q-table loaded! ({len(self.Q)} entries)")


# ── Smart Opponent ────────────────────────────────────────────────────
def smart_opponent_move(board_tuple):
    env = TicTacToe()
    board = list(board_tuple)
    empty = [i for i, x in enumerate(board) if x == 0]

    # Try to win
    for move in empty:
        env.board = tuple(board)
        env.make_move(move, -1)
        if env.check_winner(-1):
            return move

    # Try to block
    for move in empty:
        env.board = tuple(board)
        env.make_move(move, 1)
        if env.check_winner(1):
            return move

    # Take center
    if 4 in empty:
        return 4

    # Take corners
    corners = [m for m in [0, 2, 6, 8] if m in empty]
    if corners:
        return random.choice(corners)

    # Take edges
    return random.choice(empty)


# ── Training ──────────────────────────────────────────────────────────
def train(episodes=500000):
    agent = QLearningAgent()
    env = TicTacToe()

    wins = draws = losses = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        # Sometimes let opponent go first so AI learns defense
        if random.random() < 0.5:
            opp = smart_opponent_move(state)
            env.make_move(opp, -1)
            state = env.board

        while not done:
            valid = env.get_valid()
            if not valid:
                break

            action = agent.select_action(state, valid)
            env.make_move(action, 1)
            next_state = env.board

            if env.check_winner(1):
                agent.update(state, action, +10, next_state, True)
                wins += 1
                done = True

            elif env.is_full():
                agent.update(state, action, +3, next_state, True)
                draws += 1
                done = True

            else:
                # Opponent move
                opp = smart_opponent_move(next_state)
                env.make_move(opp, -1)
                next_state = env.board

                if env.check_winner(-1):
                    agent.update(state, action, -10, next_state, True)
                    losses += 1
                    done = True

                elif env.is_full():
                    agent.update(state, action, +3, next_state, True)
                    draws += 1
                    done = True

                else:
                    agent.update(state, action, 0, next_state, False)

            state = next_state

        # Linear epsilon decay
        agent.epsilon = max(0.01, 1.0 - episode / (episodes * 0.8))

        if episode % 50000 == 0:
            total = wins + draws + losses
            if total > 0:
                print(f"Ep {episode:>7} | ε={agent.epsilon:.3f} | "
                      f"W={wins/total*100:.1f}% "
                      f"D={draws/total*100:.1f}% "
                      f"L={losses/total*100:.1f}% | "
                      f"Q-table size: {len(agent.Q)}")
            wins = draws = losses = 0

    agent.save("qtable.pkl")
    return agent


# ── Play ──────────────────────────────────────────────────────────────
def play(agent):
    env = TicTacToe()
    print("\n=== Tic-Tac-Toe ===")
    print("You are O (-1), AI is X (1)")
    print("Positions:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 \n")

    state = env.reset()
    done = False

    while not done:
        env.print_board()
        valid = env.get_valid()

        # Human move
        while True:
            try:
                move = int(input(f"Your move {valid}: "))
                if move in valid:
                    break
                print("❌ Invalid move, try again.")
            except ValueError:
                print("❌ Enter a number 0-8.")

        env.make_move(move, -1)
        state = env.board

        if env.check_winner(-1):
            env.print_board()
            print("🎉 You win!")
            done = True
            break

        if env.is_full():
            env.print_board()
            print("🤝 Draw!")
            done = True
            break

        # AI move
        valid = env.get_valid()
        ai_action = agent.get_best_action(state, valid)  # always best action
        print(f"\n🤖 AI plays: {ai_action}")
        env.make_move(ai_action, 1)
        state = env.board

        if env.check_winner(1):
            env.print_board()
            print("🤖 AI wins!")
            done = True
            break

        if env.is_full():
            env.print_board()
            print("🤝 Draw!")
            done = True
            break

    if input("\nPlay again? (y/n): ").lower() == "y":
        play(agent)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    if os.path.exists("qtable.pkl"):
        choice = input("Found saved Q-table. Load it? (y/n): ")
        agent = QLearningAgent(epsilon=0.0)
        if choice.lower() == "y":
            agent.load("qtable.pkl")
        else:
            print("Training from scratch...")
            agent = train()
    else:
        print("No saved model found. Training...")
        agent = train()

    agent.epsilon = 0.0  # no exploration when playing
    play(agent)