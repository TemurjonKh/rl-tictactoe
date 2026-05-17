import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
import os

# ── Environment ───────────────────────────────────────────────────────
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        # Always from current player's perspective → key AlphaZero idea
        return self.board * self.current_player

    def get_valid(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == 0]

    def get_valid_mask(self):
        mask = np.zeros(9, dtype=np.float32)
        for r, c in self.get_valid():
            mask[r * 3 + c] = 1
        return mask

    def make_move(self, action):
        r, c = action // 3, action % 3
        if self.board[r][c] != 0:
            return None, False
        self.board[r][c] = self.current_player
        done = self.check_winner(self.current_player) or self.is_full()
        self.current_player *= -1
        return self.get_state(), done

    def check_winner(self, player):
        b = self.board
        for i in range(3):
            if np.all(b[i, :] == player): return True
            if np.all(b[:, i] == player): return True
        if np.all(np.diag(b) == player): return True
        if np.all(np.diag(np.fliplr(b)) == player): return True
        return False

    def get_result(self, player):
        if self.check_winner(player):   return +1
        if self.check_winner(-player):  return -1
        if self.is_full():              return 0
        return None  # game not over

    def is_full(self):
        return len(self.get_valid()) == 0

    def clone(self):
        clone = TicTacToe()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        return clone

    def print_board(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print()
        for r in range(3):
            print(" | ".join(symbols[int(self.board[r][c])] for c in range(3)))
            if r < 2: print("---------")
        print()


# ── Neural Network (Policy + Value heads) ─────────────────────────────
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared body
        self.shared = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy head → probability of each move
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9),    # 9 possible moves
        )

        # Value head → how good is this position (-1 to +1)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()            # output between -1 and +1
        )

    def forward(self, x):
        shared = self.shared(x)
        policy = self.policy_head(shared)   # raw logits
        value  = self.value_head(shared)    # scalar -1 to +1
        return policy, value


# ── MCTS Node ─────────────────────────────────────────────────────────
class MCTSNode:
    def __init__(self, env, parent=None, action=None, prior=0.0):
        self.env = env                  # game state at this node
        self.parent = parent            # parent node
        self.action = action            # action that led here
        self.prior = prior              # P(action) from neural net

        self.children = {}              # action → MCTSNode
        self.visit_count = 0            # N
        self.value_sum = 0.0            # W (total value)

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count   # Q = W/N

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, parent_visits, c_puct=1.5):
        # UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)
        # Balances exploitation (Q) vs exploration (P term)
        exploration = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration


# ── MCTS ──────────────────────────────────────────────────────────────
class MCTS:
    def __init__(self, net, simulations=100, c_puct=1.5):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct

    def search(self, env, temperature=1.0):
        root = MCTSNode(env.clone())

        # Add Dirichlet noise to root for exploration
        self._expand(root)
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.simulations):
            node = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        # Build move probabilities from visit counts
        move_probs = np.zeros(9)
        for action, child in root.children.items():
            move_probs[action] = child.visit_count

        if move_probs.sum() > 0:
            if temperature == 0:
                # Greedy — pick most visited
                best = np.argmax(move_probs)
                move_probs = np.zeros(9)
                move_probs[best] = 1.0
            else:
                # Softmax with temperature
                move_probs = move_probs ** (1.0 / temperature)
                move_probs /= move_probs.sum()

        return move_probs

    def _select(self, node):
        # Walk down tree picking highest UCB until leaf
        while not node.is_leaf():
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = child.ucb_score(node.visit_count, self.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child

        # Check if game is over
        result = node.env.get_result(node.env.current_player * -1)
        if result is not None:
            return node  # terminal node

        self._expand(node)
        # Pick first child to evaluate
        if node.children:
            node = random.choice(list(node.children.values()))
        return node

    def _expand(self, node):
        # Ask neural net for policy and value
        state = node.env.get_state().flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        valid_mask = node.env.get_valid_mask()

        with torch.no_grad():
            policy_logits, value = self.net(state_tensor)

        # Mask invalid moves and softmax
        policy = policy_logits.squeeze().numpy()
        policy = np.where(valid_mask == 1, policy, -1e9)
        policy = np.exp(policy - policy.max())
        policy *= valid_mask
        if policy.sum() > 0:
            policy /= policy.sum()

        # Create child nodes
        for action in range(9):
            if valid_mask[action] == 1:
                child_env = node.env.clone()
                child_env.make_move(action)
                node.children[action] = MCTSNode(
                    child_env,
                    parent=node,
                    action=action,
                    prior=policy[action]
                )

        node._value = value.item()

    def _evaluate(self, node):
        result = node.env.get_result(node.env.current_player * -1)
        if result is not None:
            return result  # terminal → exact value
        return node._value if hasattr(node, "_value") else 0.0

    def _backpropagate(self, node, value):
        # Walk back up flipping value (what's good for one player is bad for other)
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value   # flip for opponent's perspective
            node = node.parent

    def _add_dirichlet_noise(self, root, alpha=0.8, epsilon=0.25):
        # Adds randomness to root to ensure exploration
        if not root.children:
            return
        noise = np.random.dirichlet([alpha] * len(root.children))
        for (action, child), n in zip(root.children.items(), noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * n


# ── Replay Buffer ─────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ── AlphaZero Trainer ─────────────────────────────────────────────────
class AlphaZeroTrainer:
    def __init__(self, simulations=100):
        self.net = AlphaZeroNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.mcts = MCTS(self.net, simulations=simulations)
        self.buffer = ReplayBuffer()

    def self_play(self, temperature_threshold=6):
        env = TicTacToe()
        env.reset()
        game_history = []  # (state, policy, current_player)

        step = 0
        while True:
            valid = env.get_valid()
            if not valid:
                break

            # High temperature early → more exploration
            # Low temperature later → more exploitation
            temp = 1.0 if step < temperature_threshold else 0.1

            move_probs = self.mcts.search(env, temperature=temp)

            # Save state and policy
            state = env.get_state().flatten()
            game_history.append((state, move_probs, env.current_player))

            # Sample action from probabilities
            action = np.random.choice(9, p=move_probs)
            env.make_move(action)
            step += 1

            # Check game over
            result = env.get_result(env.current_player * -1)
            if result is not None:
                # Label every position with outcome from that player's perspective
                for state, policy, player in game_history:
                    value = result if player == env.current_player * -1 else -result
                    self.buffer.push(state, policy, value)
                return result

        return 0

    def train_network(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        states, policies, values = zip(*batch)

        states   = torch.FloatTensor(np.array(states))
        policies = torch.FloatTensor(np.array(policies))
        values   = torch.FloatTensor(np.array(values)).unsqueeze(1)

        policy_logits, value_pred = self.net(states)

        # Policy loss: cross entropy between MCTS probs and network probs
        policy_loss = -(policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

        # Value loss: how wrong was our position evaluation
        value_loss = nn.MSELoss()(value_pred, values)

        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()

    def train(self, iterations=200, self_play_games=20, batch_size=64):
        print("🧠 Training AlphaZero...\n")
        wins = draws = losses = 0

        for iteration in range(1, iterations + 1):

            # Self play
            for _ in range(self_play_games):
                result = self.self_play()
                if result == 1:  wins += 1
                elif result == 0: draws += 1
                else:            losses += 1

            # Train network
            loss = self.train_network(batch_size)

            if iteration % 10 == 0:
                total = wins + draws + losses
                print(f"Iter {iteration:>4} | "
                      f"Loss={loss:.4f} if loss else 'N/A' | "
                      f"W={wins/total*100:.1f}% "
                      f"D={draws/total*100:.1f}% "
                      f"L={losses/total*100:.1f}% | "
                      f"Buffer: {len(self.buffer)}")
                wins = draws = losses = 0

        torch.save(self.net.state_dict(), "alphazero_ttt.pth")
        print("\n✅ Model saved!")

    def load(self):
        self.net.load_state_dict(torch.load("alphazero_ttt.pth"))
        self.net.eval()
        print("✅ Model loaded!")


# ── Play ──────────────────────────────────────────────────────────────
def play(trainer):
    env = TicTacToe()
    env.reset()
    print("\n=== Tic-Tac-Toe (AlphaZero) ===")
    print("You are O, AI is X")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 \n")

    done = False
    while not done:
        env.print_board()
        valid_flat = [r * 3 + c for r, c in env.get_valid()]

        # Human move
        while True:
            try:
                move = int(input(f"Your move {valid_flat}: "))
                if move in valid_flat:
                    break
                print("❌ Invalid move.")
            except ValueError:
                print("❌ Enter a number.")

        env.make_move(move)

        result = env.get_result(env.current_player * -1)
        if result is not None:
            env.print_board()
            if result == -1: print("🎉 You win!")
            elif result == 0: print("🤝 Draw!")
            break

        # AI move — temperature=0 means always pick best move
        print("\n🤖 AI is thinking...")
        move_probs = trainer.mcts.search(env, temperature=0)
        ai_action = np.argmax(move_probs)
        print(f"🤖 AI plays: {ai_action}")
        env.make_move(ai_action)

        result = env.get_result(env.current_player * -1)
        if result is not None:
            env.print_board()
            if result == 1: print("🤖 AI wins!")
            elif result == 0: print("🤝 Draw!")
            break

    if input("\nPlay again? (y/n): ").lower() == "y":
        play(trainer)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = AlphaZeroTrainer(simulations=100)

    if os.path.exists("alphazero_ttt.pth"):
        choice = input("Found saved model. Load it? (y/n): ")
        if choice.lower() == "y":
            trainer.load()
        else:
            trainer.train(iterations=200, self_play_games=20)
    else:
        trainer.train(iterations=200, self_play_games=20)

    play(trainer)