from agent import Agent
from oxono import Game
import math
import random
random.seed(0)
import time

class V1(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.max_depth = 3
        self.pv_cache = {}
        self.eval_cache = {} 
        self.start_cpu_time = 0
        self.time_limit = 0

    def act(self, state, remaining_time):
        legal_actions = Game.actions(state)
        if not legal_actions:
            return None

        self.start_cpu_time = time.time()

        empty_cells = sum(1 for r in range(6) for c in range(6) if state.board[r][c] is None)
        estimated_moves_left = max(4.0, empty_cells / 2.0)
        ideal_time = remaining_time / estimated_moves_left
        self.time_limit = max(0.2, min(ideal_time, remaining_time * 0.15) - 0.05)

        sorted_initial_actions = self._sort_actions(state, legal_actions, True)
        best_move_so_far = sorted_initial_actions[0]
        
        for current_depth in range(1, 20):
            try:
                score, move = self.max_value(state, current_depth, -math.inf, math.inf)
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
        
        if depth == 0 or Game.is_terminal(state):
            return self.evaluate(state), None
        
        state_hash = str(state.board)
        cached_move = self.pv_cache.get(state_hash)

        sorted_actions = self._sort_actions(state, Game.actions(state), True, priority_move=cached_move)
        
        v = -math.inf
        best_move = None
        
        for action in sorted_actions:
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
                break
            
        self.pv_cache[state_hash] = best_move
        return v, best_move
    
    def min_value(self, state, depth, alpha, beta):
        if (time.time() - self.start_cpu_time) > self.time_limit:
            raise TimeoutError
        
        if depth == 0 or Game.is_terminal(state):
            return self.evaluate(state), None
        
        state_hash = str(state.board)
        cached_move = self.pv_cache.get(state_hash)

        sorted_actions = self._sort_actions(state, Game.actions(state), False, priority_move=cached_move)

        v = math.inf
        best_move = None
        
        for action in sorted_actions:
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
                break 
                
        self.pv_cache[state_hash] = best_move
        return v, best_move
    
    def _sort_actions(self, state, actions, maximizing_player, priority_move=None):
        actions_copy = list(actions)
        random.shuffle(actions_copy)

        def quick_score(action):
            pos = action[1] 
            r, c = pos[0], pos[1]
            if r in [2, 3] and c in [2, 3]: return 10
            if r in [1, 4] and c in [1, 4]: return 5
            return 0
        
        actions_copy.sort(key=quick_score, reverse=True)

        if priority_move is not None and priority_move in actions_copy:
            actions_copy.remove(priority_move)
            actions_copy.insert(0, priority_move)

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
                window_row = row[j:j+4]
                window_col = col[j:j+4]
                score += self._score_window(window_row)
                score += self._score_window(window_col)

        for r in range(3):
            for c in range(3): 
                window_diag1 = [board[r+i][c+i] for i in range(4)]
                window_diag2 = [board[r+3-i][c+i] for i in range(4)]
                score += self._score_window(window_diag1)
                score += self._score_window(window_diag2)

        self.eval_cache[state_hash] = score
        return score
    
    def _score_window(self, window):
        score = 0
        pieces = [cell for cell in window if cell is not None]
        empty_count = 4 - len(pieces)
        
        if empty_count == 4:
            return 0
            
        colors = [cell[1] for cell in pieces]
        symbols = [cell[0] for cell in pieces]

        my_color_count = colors.count(self.player)
        opp_color_count = colors.count(1 - self.player)
        
        if my_color_count > 0 and opp_color_count == 0:
            if my_color_count == 3: score += 50
            elif my_color_count == 2: score += 10 

        elif opp_color_count > 0 and my_color_count == 0:
            if opp_color_count == 3: score -= 60
            elif opp_color_count == 2: score -= 12

        x_count = symbols.count('x')
        o_count = symbols.count('o')
        
        if x_count == 3 and o_count == 0: score += 20 
        elif o_count == 3 and x_count == 0: score += 20

        return score