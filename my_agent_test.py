import math
import random
import time
from agent import Agent
from oxono import Game

class MyAgentTest(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.cache = {} 
        self.start_cpu_time = 0
        self.time_limit = 0

    def act(self, state, remaining_time):
        legal_actions = Game.actions(state)
        if not legal_actions: return None

        self.start_cpu_time = time.time()
        
        total_pieces = 32 - (sum(state.pieces_x) + sum(state.pieces_o))
        my_turns_left = max(1, (32 - total_pieces) // 2)
        
        safe_remaining_time = remaining_time - 50 
        base_time = max(0.5, safe_remaining_time / my_turns_left)

        if total_pieces < 8:
            self.time_limit = base_time * 0.7
        elif total_pieces < 24:
            self.time_limit = base_time * 1.3 
        else:
            self.time_limit = base_time * 0.8 

        self.time_limit = min(self.time_limit, 15.0)

        best_move_so_far = legal_actions[0]
        last_score = 0

        for current_depth in range(1, 25):
            try:
                score, move = self.max_value(state, current_depth, -math.inf, math.inf)
                if move:
                    best_move_so_far = move
                    last_score = score
                
                if abs(score) >= 1000: break 
                
                if current_depth >= 6 and (time.time() - self.start_cpu_time) > (self.time_limit * 0.5):
                    break

            except TimeoutError:
                break
        
        return best_move_so_far

    def _get_state_hash(self, state):
        return hash((str(state.board), state.totem_O, state.totem_X, state.current_player))
    
    def max_value(self, state, depth, alpha, beta):
        if (time.time() - self.start_cpu_time) > self.time_limit: raise TimeoutError
        
        state_hash = self._get_state_hash(state)
        if state_hash in self.cache:
            val, d = self.cache[state_hash]
            if d >= depth: return val, None

        if depth == 0 or Game.is_terminal(state):
            return self.evaluate(state), None
        
        v = -math.inf
        best_move = None
        for action in self._sort_actions(state, Game.actions(state), True):
            new_state = state.copy()
            Game.apply(new_state, action)
            v2, _ = (self.max_value(new_state, depth-1, alpha, beta) if Game.to_move(new_state) == self.player 
                     else self.min_value(new_state, depth-1, alpha, beta))
            if v2 > v:
                v, best_move = v2, action
                alpha = max(alpha, v)
            if v >= beta: break
        
        self.cache[state_hash] = (v, depth)
        return v, best_move

    def min_value(self, state, depth, alpha, beta):
        if (time.time() - self.start_cpu_time) > self.time_limit: raise TimeoutError
        
        state_hash = self._get_state_hash(state)
        if state_hash in self.cache:
            val, d = self.cache[state_hash]
            if d >= depth: return val, None

        if depth == 0 or Game.is_terminal(state):
            return self.evaluate(state), None
        
        v = math.inf
        best_move = None
        for action in self._sort_actions(state, Game.actions(state), False):
            new_state = state.copy()
            Game.apply(new_state, action)
            v2, _ = (self.max_value(new_state, depth-1, alpha, beta) if Game.to_move(new_state) == self.player 
                     else self.min_value(new_state, depth-1, alpha, beta))
            if v2 < v:
                v, best_move = v2, action
                beta = min(beta, v)
            if v <= alpha: break
                
        self.cache[state_hash] = (v, depth)
        return v, best_move

    def _sort_actions(self, state, actions, maximizing_player):
        def quick_score(a):
            r, c = a[2][0], a[2][1]
            score = 0
            if r in [2, 3] and c in [2, 3]: score = 10
            elif r in [1, 4] and c in [1, 4]: score = 5
            return score
        return sorted(actions, key=quick_score, reverse=maximizing_player)

    def evaluate(self, state):
        state_hash = self._get_state_hash(state)
        if state_hash in self.cache:
            val, d = self.cache[state_hash]
            return val
            
        if Game.is_terminal(state):
            u = Game.utility(state, self.player)
            return 2000 if u == 1 else (-2000 if u == -1 else 0)
                
        score = 0
        board = state.board
 
        for i in range(6):
            row = board[i]
            col = [board[j][i] for j in range(6)]
            for j in range(3):
                score += self._score_window(row[j:j+4])
                score += self._score_window(col[j:j+4])

        tr, tc = state.totem_O if self.player == 0 else state.totem_X
        if tr in [0, 5] or tc in [0, 5]: score -= 15
        
        return score
    
    def _score_window(self, window):
        pieces = [c for c in window if c is not None]
        if not pieces: return 0
        
        my_p = [p for p in pieces if p[1] == self.player]
        opp_p = [p for p in pieces if p[1] != self.player]
        
        m_c = len(my_p)
        o_c = len(opp_p)

        if o_c > 0 and m_c == 0:
            if o_c == 3: return -250
            if o_c == 2: return -50
            if o_c == 1: return -5

        if m_c > 0 and o_c == 0:
            if m_c == 3: return 150
            if m_c == 2: return 40
            if m_c == 1: return 10

        symbols = [p[0] for p in pieces]
        for s in ['x', 'o']:
            s_c = symbols.count(s)
            if s_c == 3 and len(pieces) == 3:
                return 70
                
        return 0