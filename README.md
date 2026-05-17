# Reinforcement Learning on Tic-Tac-Toe
Comparative study of three RL approaches implemented from scratch

## What I built and why
I wanted to understand WHY different RL methods exist
and WHEN to use each one. TTT served as a simple
verifiable sandbox to implement and compare:

| Method      | Win Rate | Training Time | Notes                    |
|-------------|----------|---------------|--------------------------|
| DQN         | 71%      | ~10 min       | Struggled with blocking  |
| Q-learning  | 98%      | ~2 min        | Perfect for small states |
| AlphaZero   | 99%      | ~5 min        | Overkill but educational |

## Key learnings
- DQN needs reward shaping to learn blocking behavior
- Q-learning outperforms DQN on small state spaces (5478 states)
- AlphaZero's MCTS + self-play needs no reward engineering at all
- Minimax is not AI - it is just a searching code

## Architecture
### AlphaZero
- Dual headed neural network (policy + value)
- MCTS with UCB selection and Dirichlet noise
- Self play training - no hardcoded opponent

### Q-learning
- Q-table with optimistic initialization
- Smart opponent for training
- Reward shaping for strategic positions

### DQN
- Experience replay buffer
- Target network for stability
- Epsilon greedy exploration

## How to run
pip install -r requirements.txt
python AlphaZero.py --agent alphazero
python DQN.py --agent DQN
python Q-learning.py --agent Q-learning


## What I would do differently
- Extend to Connect4 where AlphaZero truly shines
- Add Elo rating system to compare agents against each other
- Implement proper hyperparameter tuning
