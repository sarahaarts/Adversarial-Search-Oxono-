from agent import Agent
from oxono import Game
import math
import random
import time

class V2(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.max_depth = 3
        self.pv_cache = {}
        self.eval_cache = {}
        self.start_cpu_time = 0
        self.time_limit = 0

        self.zobrist_table = [[[random.getrandbits(64) for _ in range(4)] for _ in range(6)] for _ in range(6)]
        self.zobrist_player = random.getrandbits(64)

        self.tt = {}
        self.killer_moves = {}

    def piece_index(self, cell):
        if cell is None:
            return 0
        symbol, player = cell
        if symbol == 'x' and player == 0:
            return 1
        if symbol == 'o' and player == 0:
            return 2
        if symbol == 'x' and player == 1:
            return 3
        if symbol == 'o' and player == 1:
            return 4

    def compute_hash(self, state):
        h = 0
        for r in range(6):
            for c in range(6):
                idx = self.piece_index(state.board[r][c])
                if idx != 0:
                    h ^= self.zobrist_table[r][c][idx-1]

        if Game.to_move(state) == self.player:
            h ^= self.zobrist_player

        return h

    def act(self, state, remaining_time):
        self.tt = {} 
        self.killer_moves = {}

        legal_actions = Game.actions(state)
        if not legal_actions:
            return None

        self.start_cpu_time = time.time()

        empty_cells = sum(1 for r in range(6) for c in range(6) if state.board[r][c] is None)
        estimated_moves_left = max(4.0, empty_cells / 2.0)
        ideal_time = remaining_time / estimated_moves_left
        self.time_limit = max(0.2, min(ideal_time, remaining_time * 0.15) - 0.05)

        sorted_initial_actions = self._sort_actions(state, legal_actions, True, 0)
        best_move_so_far = sorted_initial_actions[0]

        for depth in range(1, 20):
            try:
                score, move = self.max_value(state, depth, -math.inf, math.inf)
                if move:
                    best_move_so_far = move
                if score >= 1000:
                    break
            except TimeoutError:
                break

            if (time.time() - self.start_cpu_time) > self.time_limit:
                break

        return best_move_so_far

    def max_value(self, state, depth, alpha, beta):
        if (time.time() - self.start_cpu_time) > self.time_limit:
            raise TimeoutError

        state_hash = self.compute_hash(state)

        if state_hash in self.tt:
            stored_depth, stored_value, flag = self.tt[state_hash]
            if stored_depth >= depth:
                if flag == "EXACT":
                    return stored_value, None
                elif flag == "LOWERBOUND":
                    alpha = max(alpha, stored_value)
                elif flag == "UPPERBOUND":
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value, None

        if depth == 0 or Game.is_terminal(state):
            val = self.evaluate(state)
            return val, None

        alpha_orig = alpha

        cached_move = self.pv_cache.get(str(state.board))
        actions = self._sort_actions(state, Game.actions(state), True, depth, priority_move=cached_move)

        v = -math.inf
        best_move = None

        for action in actions:
            new_state = state.copy()
            Game.apply(new_state, action)

            if Game.to_move(new_state) == self.player:
                v2, _ = self.max_value(new_state, depth - 1, alpha, beta)
            else:
                v2, _ = self.min_value(new_state, depth - 1, alpha, beta)

            if v2 > v:
                v = v2
                best_move = action
                alpha = max(alpha, v)

            if v >= beta:
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []

                if action not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, action)

                if len(self.killer_moves[depth]) > 2:
                    self.killer_moves[depth].pop()
                break

        if v <= alpha_orig:
            flag = "UPPERBOUND"
        elif v >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"

        self.tt[state_hash] = (depth, v, flag)
        self.pv_cache[str(state.board)] = best_move

        return v, best_move

    def min_value(self, state, depth, alpha, beta):
        if (time.time() - self.start_cpu_time) > self.time_limit:
            raise TimeoutError

        state_hash = self.compute_hash(state)

        if state_hash in self.tt:
            stored_depth, stored_value, flag = self.tt[state_hash]
            if stored_depth >= depth:
                if flag == "EXACT":
                    return stored_value, None
                elif flag == "LOWERBOUND":
                    alpha = max(alpha, stored_value)
                elif flag == "UPPERBOUND":
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value, None

        if depth == 0 or Game.is_terminal(state):
            val = self.evaluate(state)
            return val, None

        beta_orig = beta

        cached_move = self.pv_cache.get(str(state.board))
        actions = self._sort_actions(state, Game.actions(state), False, depth, priority_move=cached_move)

        v = math.inf
        best_move = None

        for action in actions:
            new_state = state.copy()
            Game.apply(new_state, action)

            if Game.to_move(new_state) == self.player:
                v2, _ = self.max_value(new_state, depth - 1, alpha, beta)
            else:
                v2, _ = self.min_value(new_state, depth - 1, alpha, beta)

            if v2 < v:
                v = v2
                best_move = action
                beta = min(beta, v)

            if v <= alpha:
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []

                if action not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, action)

                if len(self.killer_moves[depth]) > 2:
                    self.killer_moves[depth].pop()
                break

        if v <= alpha:
            flag = "UPPERBOUND"
        elif v >= beta_orig:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"

        self.tt[state_hash] = (depth, v, flag)
        self.pv_cache[str(state.board)] = best_move

        return v, best_move

    def _sort_actions(self, state, actions, maximizing_player, depth, priority_move=None):
        actions_copy = list(actions)
        random.shuffle(actions_copy)

        def quick_score(action):
            r, c = action[1]

            # centre
            score = 0
            if r in [2, 3] and c in [2, 3]: score += 5
            elif r in [1, 4] and c in [1, 4]: score += 2

            # simulation rapide
            new_state = state.copy()
            Game.apply(new_state, action)

            if Game.is_terminal(new_state):
                return 1000

            score += self.evaluate(new_state) * 0.1  # léger poids

            return score

        actions_copy.sort(key=quick_score, reverse=True)

        if priority_move is not None and priority_move in actions_copy:
            actions_copy.remove(priority_move)
            actions_copy.insert(0, priority_move)

        killers = self.killer_moves.get(depth, [])

        for killer in reversed(killers):
            if killer in actions_copy:
                actions_copy.remove(killer)
                actions_copy.insert(0, killer)

        return actions_copy

    def evaluate(self, state):
        state_hash = str(state.board)

        if state_hash in self.eval_cache:
            return self.eval_cache[state_hash]

        if Game.is_terminal(state):
            utility = Game.utility(state, self.player)
            if utility == 1:
                return 1000
            elif utility == -1:
                return -1000
            else:
                return 0

        score = 0
        board = state.board

        for r in range(6):
            for c in range(6):
                if board[r][c] is not None:
                    if board[r][c][1] == self.player:
                        if r in [2, 3] and c in [2, 3]: score += 5
                        elif r in [1, 4] and c in [1, 4]: score += 2

        for i in range(6):
            row = board[i]
            col = [board[j][i] for j in range(6)]
            for j in range(3):
                score += self._score_window(row[j:j+4])
                score += self._score_window(col[j:j+4])

        for r in range(3):
            for c in range(3):
                diag1 = [board[r+i][c+i] for i in range(4)]
                diag2 = [board[r+3-i][c+i] for i in range(4)]
                score += self._score_window(diag1)
                score += self._score_window(diag2)

        self.eval_cache[state_hash] = score
        return score

    def _score_window(self, window):
        score = 0
        pieces = [cell for cell in window if cell is not None]

        if len(pieces) == 0:
            return 0

        colors = [cell[1] for cell in pieces]
        symbols = [cell[0] for cell in pieces]

        my = colors.count(self.player)
        opp = colors.count(1 - self.player)

        if my > 0 and opp == 0:
            if my == 3: score += 50
            elif my == 2: score += 10

        elif opp > 0 and my == 0:
            if opp == 3: score -= 60
            elif opp == 2: score -= 12

        x_count = symbols.count('x')
        o_count = symbols.count('o')

        if x_count == 3 and o_count == 0: score += 20
        elif o_count == 3 and x_count == 0: score += 20

        return score