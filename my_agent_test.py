import math
import random
import time
from agent import Agent
from oxono import Game

class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.cache = {} 
        self.start_time = 0
        self.time_limit = 0

    def act(self, state, remaining_time):
        self.start_time = time.time()
        # On utilise environ 4% du temps restant par coup pour être sûr
        self.time_limit = remaining_time * 0.04
        
        legal_actions = Game.actions(state)
        if not legal_actions: return None

        best_move = legal_actions[0]
        
        # Approfondissement itératif
        for depth in range(1, 15):
            try:
                # Appel de la fonction de recherche avec Alpha-Beta
                val, move = self._search(state, depth, -1000000, 1000000, True)
                if move:
                    best_move = move
                if val >= 900000 or val <= -900000: # Victoire/Défaite forcée trouvée
                    break
            except TimeoutError:
                break
        
        return best_move

    def _get_hash(self, state):
        # Utiliser des tuples est beaucoup plus rapide que str(board)
        board_tuple = tuple(tuple(row) for row in state.board)
        return (board_tuple, state.totem_O, state.totem_X, state.current_player)

    def _search(self, state, depth, alpha, beta, maximizing):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        state_hash = self._get_hash(state)
        if state_hash in self.cache:
            res_val, res_depth = self.cache[state_hash]
            if res_depth >= depth:
                return res_val, None

        if depth == 0 or Game.is_terminal(state):
            return self.evaluate(state), None

        actions = Game.actions(state)
        # ORDONNANCEMENT : On trie pour tester les meilleures cases (centre) en premier
        # Cela multiplie l'efficacité de l'élagage Alpha-Beta
        actions.sort(key=lambda a: (abs(a[2][0]-2.5) + abs(a[2][1]-2.5)))

        best_action = None
        if maximizing:
            v = -1000001
            for a in actions:
                # --- TECHNIQUE UNDO (Pas de copie !) ---
                old_info = self._apply_and_save(state, a)
                
                # Vérifier si on rejoue (cas où l'adversaire n'a plus de pièces)
                is_next_me = (Game.to_move(state) == self.player)
                score, _ = self._search(state, depth - 1, alpha, beta, is_next_me)
                
                self._undo(state, a, old_info)
                # ----------------------------------------

                if score > v:
                    v = score
                    best_action = a
                alpha = max(alpha, v)
                if v >= beta: break
        else:
            v = 1000001
            for a in actions:
                old_info = self._apply_and_save(state, a)
                is_next_me = (Game.to_move(state) == self.player)
                score, _ = self._search(state, depth - 1, alpha, beta, is_next_me)
                self._undo(state, a, old_info)

                if score < v:
                    v = score
                    best_action = a
                beta = min(beta, v)
                if v <= alpha: break

        self.cache[state_hash] = (v, depth)
        return v, best_action

    def _apply_and_save(self, state, action):
        """Applique le coup et sauvegarde l'état précédent pour l'Undo."""
        info = (state.totem_O, state.totem_X, state.current_player, state.last_move, 
                state.pieces_x[:], state.pieces_o[:])
        Game.apply(state, action)
        return info

    def _undo(self, state, action, info):
        """Annule le coup sur l'objet state directement (Gain de temps énorme)."""
        r, c = action[2]
        state.board[r][c] = None
        state.totem_O, state.totem_X, state.current_player, state.last_move, state.pieces_x, state.pieces_o = info

    def evaluate(self, state):
        if Game.is_terminal(state):
            u = Game.utility(state, self.player)
            if u == 1: return 999999
            if u == -1: return -999999
            return 0

        score = 0
        board = state.board
        
        # On scanne les lignes et colonnes
        for i in range(6):
            score += self._score_line([board[i][j] for j in range(6)]) # Ligne
            score += self._score_line([board[j][i] for j in range(6)]) # Colonne
            
        return score

    def _score_line(self, line):
        score = 0
        for i in range(3): # Fenêtres de 4
            window = line[i:i+4]
            # --- Analyse COULEUR ---
            mine = sum(1 for p in window if p and p[1] == self.player)
            opp = sum(1 for p in window if p and p[1] != self.player)
            
            if opp == 0:
                if mine == 3: score += 800  # Menace de gagne imminente
                elif mine == 2: score += 100
            elif mine == 0:
                if opp == 3: score -= 2000 # DANGER : On doit bloquer absolument
                elif opp == 2: score -= 150

            # --- Analyse SYMBOLE (Très important en Oxono) ---
            for sym in ['x', 'o']:
                sym_count = sum(1 for p in window if p and p[0] == sym)
                if sym_count == 3 and window.count(None) == 1:
                    score += 400 # Opportunité/Danger de symbole
        return score