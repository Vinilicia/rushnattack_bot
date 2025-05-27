import retro
import cv2
import numpy as np
import time
import os
import pickle # Para salvar/carregar a tabela Q

# Intervalo de tempo entre frames (aproximado)
dt = 0 # Para treinamento, considere remover ou reduzir o time.sleep

# Lista global para armazenar os rastreadores de inimigos
enemy_trackers = []
next_tracker_id = 0

# Constantes para tipos de inimigos
ENEMY_TYPE_1 = 1
ENEMY_TYPE_2 = 2
ENEMY_TYPE_3 = 3
ENEMY_TYPE_MINE = 4

# Nome do arquivo para o save state do emulador
SAVE_STATE_FILENAME = "rushnattack_mines_state.sav"
LOAD_STATE_AT_STARTUP = False

# --- Parâmetros do Q-learning ---
ALPHA = 0.1  # Taxa de aprendizado
GAMMA = 0.99 # Fator de desconto
EPSILON = 1.0 # Taxa de exploração inicial
EPSILON_DECAY = 0.9995 # Taxa de decaimento do epsilon por episódio
MIN_EPSILON = 0.05 # Epsilon mínimo
q_table = {}

Q_TABLE_FILENAME = "rushnattack_q_table.pkl"
LOAD_Q_TABLE_AT_STARTUP = True
SAVE_Q_TABLE_PERIODICALLY = True
SAVE_Q_TABLE_EPISODE_INTERVAL = 10 # Salvar a cada X episódios

# --- Discretização do Espaço de Estados ---
PLAYER_X_BINS = np.linspace(0, 960, 15)  # Posição X do jogador (mais granularidade)
ENEMY_DX_BINS = np.linspace(-250, 250, 10) # Distância X relativa ao inimigo
ENEMY_DY_BINS = np.linspace(-150, 150, 8)  # Distância Y relativa ao inimigo (menos variação vertical)
# Adiciona um "tipo" para "nenhum inimigo próximo significativo"
ENEMY_TYPE_CATEGORIES = sorted([ENEMY_TYPE_1, ENEMY_TYPE_2, ENEMY_TYPE_3, ENEMY_TYPE_MINE])
# O último bin é para "nenhum inimigo" ou tipo desconhecido
ENEMY_TYPE_BINS = np.append(np.array(ENEMY_TYPE_CATEGORIES) - 0.5, max(ENEMY_TYPE_CATEGORIES) + 0.5)


# --- Definição do Espaço de Ações Discretas ---
# 0: Direita, 1: Esquerda, 2: Atacar, 3: Pular,
# 4: Direita+Atacar, 5: Direita+Pular, 6: Baixo, 7: Baixo+Atacar, 8: No-Op
NUM_DISCRETE_ACTIONS = 4
def get_env_action(discrete_action_index, num_buttons):
    env_action = np.zeros(num_buttons, dtype=np.uint8)
    if discrete_action_index == 0: # Direita
        env_action[7] = 1
    elif discrete_action_index == 1: # Esquerda
        env_action[6] = 1
    elif discrete_action_index == 2: # Atacar (B)
        env_action[0] = 1
    # Ação 8: No-Op (não faz nada) - útil para o agente aprender a esperar
    return env_action

class EnemyTracker:
    def __init__(self, initial_measurement, enemy_type, tracker_id):
        self.id = tracker_id
        self.type = enemy_type
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.diag([0.25, 0.25, 1.0, 1.0]).astype(np.float32) # Q
        self.kf.measurementNoiseCov = np.diag([25.0, 25.0]).astype(np.float32) # R
        initial_x = float(initial_measurement[0])
        initial_y = float(initial_measurement[1])
        self.kf.statePost = np.array([[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.array([[25.0, 0.0,  0.0,    0.0],
                                         [0.0, 25.0,  0.0,    0.0],
                                         [0.0,  0.0, 1000.0,  0.0],
                                         [0.0,  0.0,  0.0, 1000.0]],np.float32)
        self.predicted_pos = (int(self.kf.statePost[0,0]), int(self.kf.statePost[1,0]))
        self.hits = 1
        self.misses = 0
        self.age = 0

    def predict(self):
        prediction = self.kf.predict()
        self.predicted_pos = (int(prediction[0, 0]), int(prediction[1, 0]))
        self.age += 1
        return self.predicted_pos

    def update(self, measurement):
        measurement_vec = np.array([measurement[0], measurement[1]], dtype=np.float32).reshape(2,1)
        self.kf.correct(measurement_vec)
        self.hits += 1
        self.misses = 0
        self.predicted_pos = (int(self.kf.statePost[0,0]), int(self.kf.statePost[1,0]))

    def increment_misses(self):
        self.misses += 1

# --- FUNÇÕES DE DETECÇÃO (sem alterações, apenas reutilizadas) ---
def detect_player(frame):
    lower_blue = np.array([110, 200, 150])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(frame, lower_blue, upper_blue)
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=10) # Reduzi iterações para possível ganho de performance
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    min_area = 500
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
    if centers:
        return centers[0]
    return (0, 0) # Retornar uma tupla consistente

def detect_enemy1(frame, block_height=20, block_width=60):
    height, width, _ = frame.shape
    white = np.array([240, 240, 240], dtype=np.uint8)
    min_white = 150 # Ajuste este valor conforme necessário
    positions = []
    y = 570
    x = 0
    while x < width:
        if x + block_width > width:
            break
        block_frame = frame[y:y+block_height, x:x+block_width]
        if block_frame.shape[0] < block_height or block_frame.shape[1] < block_width: # Checagem de segurança
            break
        mask_white = np.all(block_frame >= white, axis=2)
        sum_white = np.sum(mask_white)
        if sum_white > min_white:
            positions.append((x+30, y-10, ENEMY_TYPE_1))
            x += 60 # Pular mais para evitar detecções múltiplas do mesmo inimigo
        else:
            x += 10 # Passo menor se nada for detectado
    return positions

def detect_enemy2(frame, block_height=20, block_width=60):
    frame_resized = frame[570:590,:]
    lower_orange = np.array([5, 250, 200])
    upper_orange = np.array([7, 255, 255])
    mask = cv2.inRange(frame_resized, lower_orange, upper_orange)
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    min_area = 130 # Ajuste se necessário
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = 560 # Coordenada Y fixa baseada na ROI
                centers.append((cx, cy, ENEMY_TYPE_2))
    return centers

def detect_enemy3(frame):
    lower_yellow = np.array([15, 197, 244]) # HSV
    upper_yellow = np.array([15, 199, 248]) # HSV
    mask = cv2.inRange(frame, lower_yellow, upper_yellow)
    kernel = np.ones((10, 10), np.uint8) # Kernel maior pode ser útil
    dilated_mask = cv2.erode(mask, kernel, iterations=1) # Erode pode ajudar a separar blobs
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    min_area = 3 # Pode precisar de ajuste
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy, ENEMY_TYPE_3))
    return centers

def detect_landmine(frame_bgr, scan_center_y=645, y_margin=5, target_color_params=None, mine_filter_params=None):
    frame_height, _ = frame_bgr.shape[:2]
    if target_color_params is None:
        target_color_params = {
            'lower_bgr': np.array([20, 20, 20]), # Ajustado para ser mais genérico para minas escuras
            'upper_bgr': np.array([100, 100, 100]) # Ajustado para ser mais genérico
        }
    if mine_filter_params is None:
        mine_filter_params = {
            'min_area': 8, 'max_area': 50, # Ajustado
            'morph_open_ksize': 3, 'morph_open_iter': 1,
            'morph_close_ksize': 3, 'morph_close_iter': 1
        }
    roi_y_start = max(0, scan_center_y - y_margin)
    roi_y_end = min(frame_height, scan_center_y + y_margin)
    if roi_y_start >= roi_y_end: return []
    roi = frame_bgr[roi_y_start:roi_y_end, :]
    if roi.size == 0: return []
    target_mask = cv2.inRange(roi, target_color_params['lower_bgr'], target_color_params['upper_bgr'])
    processed_mask = target_mask
    open_ksize = mine_filter_params.get('morph_open_ksize', 0)
    open_iter = mine_filter_params.get('morph_open_iter', 0)
    if open_ksize > 0 and open_iter > 0:
        open_kernel = np.ones((open_ksize, open_ksize), np.uint8)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, open_kernel, iterations=open_iter)
    close_ksize = mine_filter_params.get('morph_close_ksize', 0)
    close_iter = mine_filter_params.get('morph_close_iter', 0)
    if close_ksize > 0 and close_iter > 0:
        close_kernel = np.ones((close_ksize, close_ksize), np.uint8)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_mines_list = []
    min_area = mine_filter_params['min_area']
    max_area = mine_filter_params['max_area']
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy_roi = int(M["m01"] / M["m00"]) # Coordenada Y dentro da ROI
                full_frame_cy = cy_roi + roi_y_start # Coordenada Y no frame completo
                detected_mines_list.append((cx, full_frame_cy, ENEMY_TYPE_MINE))
    return detected_mines_list
# --- FIM DAS FUNÇÕES DE DETECÇÃO ---

def discretize_value(value, bins):
    return np.digitize(value, bins) - 1

def get_q_state(player_pos, enemies_for_logic):
    player_x_discrete = discretize_value(player_pos[0], PLAYER_X_BINS)
    # player_y_discrete = discretize_value(player_pos[1], PLAYER_Y_BINS) # Se for adicionar

    closest_enemy_dx_discrete = discretize_value(0, ENEMY_DX_BINS) # Valor padrão (ex: meio do range)
    #closest_enemy_dy_discrete = discretize_value(0, ENEMY_DY_BINS) # Valor padrão
    closest_enemy_type_discrete = len(ENEMY_TYPE_BINS) -1 # Bin para "nenhum inimigo" ou "padrão"

    if enemies_for_logic:
        px, py = player_pos
        best_dist = float('inf')
        closest_enemy_data = None
        for ex, ey, enemy_type in enemies_for_logic:
            d = np.hypot(px - ex, py - ey)
            if d < best_dist:
                best_dist = d
                closest_enemy_data = (ex, ey, enemy_type)

        if closest_enemy_data and best_dist < 120: # Considerar apenas inimigos realmente próximos
            ex, ey, enemy_type = closest_enemy_data
            dx = ex - px
            closest_enemy_dx_discrete = discretize_value(dx, ENEMY_DX_BINS)
            # Garante que o tipo de inimigo seja mapeado para um bin válido
            type_bin_index = np.digitize(enemy_type, ENEMY_TYPE_BINS) -1
            if 0 <= type_bin_index < len(ENEMY_TYPE_BINS) -1 : # Exclui o bin "nenhum inimigo"
                 closest_enemy_type_discrete = type_bin_index
            else: # Se tipo não mapeado, usa o bin padrão
                 closest_enemy_type_discrete = len(ENEMY_TYPE_BINS) -1

    return (
        player_x_discrete,
        closest_enemy_dx_discrete,
        closest_enemy_type_discrete
    )

def calculate_reward(player_x_current, player_x_prev, score_current, score_prev, lives_current, lives_prev, done_flag, game_won_flag=False):
    reward = 0.0

    # 1. Recompensa por pontuação
    score_delta = score_current - score_prev
    reward += score_delta * 0.5 # Ajuste o multiplicador conforme necessário

    # 2. Recompensa por progresso (se não refletido na pontuação)
    progress_delta = player_x_current - player_x_prev
    if progress_delta > 10:
        reward += progress_delta * 5 # Pequena recompensa por mover para a direita
    elif progress_delta < -10: # Penalidade por recuar muito
        reward += progress_delta * 10 # Pequena recompensa por mover para a direita

    # 3. Penalidade por perder vida
    if lives_current < lives_prev:
        reward -= 100.0  # Penalidade grande

    # 4. Recompensa/Penalidade no final do jogo
    if done_flag:
        if game_won_flag: # Você precisaria de uma lógica para detectar vitória
            reward += 500.0
        elif lives_current == 0 : # Morreu e acabou o jogo
            reward -= 200.0
        # else: # Acabou por outro motivo (ex: tempo)
            # reward -= 50.0 # Penalidade moderada

    # 5. Pequena penalidade por passo para encorajar eficiência (opcional)
    reward -= 0.01
    return reward

def process_frame_and_get_detections(obs_frame):
    global enemy_trackers, next_tracker_id # Modificando globais

    obs_resized = cv2.resize(obs_frame, (960, 900))[200:900-50, :]
    frame_rgb = obs_resized.copy()
    frame_hsv = cv2.cvtColor(obs_resized, cv2.COLOR_RGB2HSV)

    current_player_pos = detect_player(frame_hsv)
    if current_player_pos == (0,0) and hasattr(process_frame_and_get_detections, 'last_known_player_pos'):
        current_player_pos = process_frame_and_get_detections.last_known_player_pos # Usa a última conhecida se não detectado
    else:
        process_frame_and_get_detections.last_known_player_pos = current_player_pos


    for tracker in enemy_trackers:
        tracker.predict()

    detections_e1 = detect_enemy1(frame_rgb)
    detections_e2 = detect_enemy2(frame_hsv)
    detections_e3 = detect_enemy3(frame_hsv)
    detections_mines = detect_landmine(frame_rgb)

    all_current_detections = []
    for x, y, type_val in detections_e1: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False})
    for x, y, type_val in detections_e2: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False})
    for x, y, type_val in detections_e3: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False})
    for x, y, type_val in detections_mines: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False})

    max_misses = 3
    association_threshold = 75
    active_trackers_this_frame = []
    for tracker in enemy_trackers:
        best_match_detection_idx = -1
        min_dist = association_threshold
        for i, det in enumerate(all_current_detections):
            if not det['matched'] and det['type'] == tracker.type:
                dist = np.hypot(tracker.predicted_pos[0] - det['pos'][0], tracker.predicted_pos[1] - det['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    best_match_detection_idx = i
        if best_match_detection_idx != -1:
            tracker.update(all_current_detections[best_match_detection_idx]['pos'])
            all_current_detections[best_match_detection_idx]['matched'] = True
            active_trackers_this_frame.append(tracker)
        else:
            tracker.increment_misses()
            if tracker.misses <= max_misses:
                active_trackers_this_frame.append(tracker)
    enemy_trackers = active_trackers_this_frame

    for det in all_current_detections:
        if not det['matched']:
            new_tracker = EnemyTracker(det['pos'], det['type'], next_tracker_id)
            enemy_trackers.append(new_tracker)
            next_tracker_id += 1

    current_enemies_for_logic = []
    for tracker in enemy_trackers:
        if tracker.misses < max_misses and tracker.age > 0: # Idade > 0 para dar tempo de uma predição
             current_enemies_for_logic.append((tracker.predicted_pos[0], tracker.predicted_pos[1], tracker.type))

    return current_player_pos, current_enemies_for_logic, frame_rgb


def main():
    global enemy_trackers, next_tracker_id, obs, q_table, EPSILON

    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    num_buttons = env.action_space.shape[0]

    # Carregar Tabela Q
    if LOAD_Q_TABLE_AT_STARTUP and os.path.exists(Q_TABLE_FILENAME):
        print(f"Carregando Tabela Q de {Q_TABLE_FILENAME}...")
        try:
            with open(Q_TABLE_FILENAME, "rb") as f:
                q_table = pickle.load(f)
            print("Tabela Q carregada com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar Tabela Q: {e}. Começando com uma tabela vazia.")
            q_table = {}
    else:
        q_table = {}
        if LOAD_Q_TABLE_AT_STARTUP:
            print(f"Arquivo da Tabela Q '{Q_TABLE_FILENAME}' não encontrado. Começando com uma tabela vazia.")

    def reset_q_learning_episode_vars(initial_obs, initial_info):
        nonlocal prev_player_x, prev_score, prev_lives, total_episode_reward

        # Processa o frame inicial para obter a posição inicial do jogador
        # p_pos, _, _ = process_frame_and_get_detections(initial_obs) # Não usar globais aqui
        # prev_player_x = p_pos[0] if p_pos else 0
        # Simplificação: assumir que o jogador começa em X=0 ou um valor fixo se a detecção falhar no reset.
        # A detecção real ocorrerá no início do loop principal.
        prev_player_x = 0 # Será atualizado corretamente após o primeiro processamento de frame

        prev_score = initial_info.get('score', 0)
        prev_lives = initial_info.get('lives', 5) # Assumindo 3 vidas iniciais se não estiver no info
        total_episode_reward = 0.0

        # Reseta rastreadores de inimigos para o novo episódio
        global enemy_trackers, next_tracker_id
        enemy_trackers = []
        next_tracker_id = 0


    # Inicializa variáveis de estado/recompensa do Q-learning
    # Info inicial pode não ter tudo, mas tentamos obter
    initial_dummy_info = {} # env.reset() não retorna info em todas as versões do gym
    # Para obter info inicial, damos um passo "dummy"
    _, _, _, initial_dummy_info = env.step(np.zeros(num_buttons, dtype=np.uint8))
    obs = env.reset() # Reset real para começar o episódio

    prev_player_x = 0 # Será atualizado no loop
    prev_score = initial_dummy_info.get('score', 0)
    prev_lives = initial_dummy_info.get('lives', 5)
    total_episode_reward = 0.0
    episode_count = 0

    reset_q_learning_episode_vars(obs, initial_dummy_info) # Chamada inicial

    # Carregar estado do emulador (opcional, para depuração)
    if LOAD_STATE_AT_STARTUP and os.path.exists(SAVE_STATE_FILENAME):
        print(f"Tentando carregar estado do emulador de {SAVE_STATE_FILENAME}...")
        try:
            with open(SAVE_STATE_FILENAME, "rb") as f:
                state_data = f.read()
            if env.em.set_state(state_data):
                print("Estado do emulador carregado com sucesso.")
                obs, _, _, initial_dummy_info = env.step(np.zeros(num_buttons, dtype=np.uint8))
                reset_q_learning_episode_vars(obs, initial_dummy_info) # Reseta vars do Q-learning com o novo estado
            else:
                print("Falha ao carregar estado do emulador.")
        except Exception as e:
            print(f"Erro ao carregar estado do emulador: {e}.")


    running = True
    while running:
        current_player_pos, current_enemies_for_logic, debug_frame_rgb = process_frame_and_get_detections(obs)

        # Atualiza prev_player_x se for o primeiro frame real após um reset onde a detecção pode ter falhado
        if prev_player_x == 0 and current_player_pos[0] != 0 : # Evita que seja sempre 0 no inicio
            prev_player_x = current_player_pos[0]

        current_q_s = get_q_state(current_player_pos, current_enemies_for_logic)

        # Escolher Ação (Epsilon-greedy)
        if np.random.rand() < EPSILON:
            action_idx = np.random.randint(0, NUM_DISCRETE_ACTIONS)  # Explorar
        else:
            q_values_for_s = q_table.get(current_q_s, np.zeros(NUM_DISCRETE_ACTIONS))
            if np.all(q_values_for_s == 0): # Se todas as ações são desconhecidas/zero, escolhe aleatoriamente
                 action_idx = np.random.randint(0, NUM_DISCRETE_ACTIONS)
            else:
                 action_idx = np.argmax(q_values_for_s)  # Explorar (melhor ação)

        env_action_to_take = get_env_action(action_idx, num_buttons)
        next_obs, _, done, info = env.step(env_action_to_take)

        # Processar o próximo estado para Q-update
        next_player_pos, next_enemies_for_logic, _ = process_frame_and_get_detections(next_obs)
        next_q_s_prime = get_q_state(next_player_pos, next_enemies_for_logic)

        # Calcular Recompensa
        current_score = info.get('score', prev_score)
        current_lives = info.get('lives', prev_lives)

        # Determinar se o jogo foi "ganho" é específico do jogo e pode precisar de lógica customizada
        game_won = False # Placeholder
        # Ex: if info.get('level_complete', False): game_won = True

        custom_reward = calculate_reward(current_player_pos[0], prev_player_x,
                                         current_score, prev_score,
                                         current_lives, prev_lives,
                                         done, game_won)
        total_episode_reward += custom_reward

        # Atualizar Tabela Q
        old_q_value = q_table.get(current_q_s, np.zeros(NUM_DISCRETE_ACTIONS))[action_idx]
        if done:
            next_max_q = 0.0 # Não há próximo estado após o terminal
        else:
            next_max_q = np.max(q_table.get(next_q_s_prime, np.zeros(NUM_DISCRETE_ACTIONS)))

        new_q_value = old_q_value + ALPHA * (custom_reward + GAMMA * next_max_q - old_q_value)

        if current_q_s not in q_table:
            q_table[current_q_s] = np.zeros(NUM_DISCRETE_ACTIONS)
        q_table[current_q_s][action_idx] = new_q_value

        # Preparar para a próxima iteração
        obs = next_obs
        prev_player_x = current_player_pos[0] # Posição X do estado que levou à recompensa
        prev_score = current_score
        prev_lives = current_lives

        # Visualização
        debug_bgr_frame = cv2.cvtColor(debug_frame_rgb, cv2.COLOR_RGB2BGR)
        px, py = current_player_pos
        cv2.rectangle(debug_bgr_frame, (px-15, py-30), (px+15, py+30), (255, 0, 0), 1) # Player BB menor

        for tracker in enemy_trackers: # Usar enemy_trackers globais que foram atualizados
            pred_x, pred_y = tracker.predicted_pos
            color = (0,0,0)
            if tracker.type == ENEMY_TYPE_1: color = (0, 0, 255)
            elif tracker.type == ENEMY_TYPE_2: color = (0, 255, 0)
            elif tracker.type == ENEMY_TYPE_3: color = (0, 255, 255)
            elif tracker.type == ENEMY_TYPE_MINE: color = (255, 0, 255)
            cv2.circle(debug_bgr_frame, (pred_x, pred_y), 5, color, -1)
            # cv2.putText(debug_bgr_frame, f"T{tracker.type}", (pred_x + 5, pred_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        cv2.putText(debug_bgr_frame, f"Ep: {episode_count} Rew: {total_episode_reward:.2f} Eps: {EPSILON:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(debug_bgr_frame, f"Lives: {current_lives} Score: {current_score}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)


        cv2.imshow("Rush'n Attack - Q-learning", debug_bgr_frame)
        key_pressed = cv2.waitKey(1) & 0xFF # Diminuir waitKey para acelerar treinamento se dt for baixo
        if key_pressed == 27: # ESC
            running = False
            break
        elif key_pressed == ord('s'): # Salvar estado do emulador
            print("Salvando estado do emulador...")
            try:
                state_data_emu = env.em.get_state()
                with open(SAVE_STATE_FILENAME, "wb") as f: f.write(state_data_emu)
                print(f"Estado do emulador salvo em {SAVE_STATE_FILENAME}")
            except Exception as e: print(f"Erro ao salvar estado do emulador: {e}")
        elif key_pressed == ord('l'): # Carregar estado do emulador
             if os.path.exists(SAVE_STATE_FILENAME):
                print(f"Carregando estado do emulador de {SAVE_STATE_FILENAME}...")
                try:
                    with open(SAVE_STATE_FILENAME, "rb") as f: state_data_emu = f.read()
                    if env.em.set_state(state_data_emu):
                        print("Estado do emulador carregado.")
                        obs, _, _, initial_dummy_info = env.step(np.zeros(num_buttons, dtype=np.uint8))
                        reset_q_learning_episode_vars(obs, initial_dummy_info) # Importante resetar com o novo estado
                    else:
                        print("Falha ao carregar estado do emulador.")
                except Exception as e: print(f"Erro ao carregar estado do emulador: {e}")
             else:
                print(f"Arquivo de save state '{SAVE_STATE_FILENAME}' não encontrado.")


        if done:
            print(f"Episódio {episode_count} finalizado. Recompensa total: {total_episode_reward:.2f}. Epsilon: {EPSILON:.4f}")
            obs = env.reset()
            _, _, _, initial_dummy_info_done = env.step(np.zeros(num_buttons, dtype=np.uint8)) # Para obter info após reset
            reset_q_learning_episode_vars(obs, initial_dummy_info_done)
            episode_count += 1

            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY

            if SAVE_Q_TABLE_PERIODICALLY and episode_count % SAVE_Q_TABLE_EPISODE_INTERVAL == 0:
                try:
                    with open(Q_TABLE_FILENAME, "wb") as f:
                        pickle.dump(q_table, f)
                    print(f"Tabela Q salva em {Q_TABLE_FILENAME} no episódio {episode_count}")
                except Exception as e:
                    print(f"Erro ao salvar Tabela Q: {e}")

        if dt > 0.001 : # Só dar sleep se dt for significativo, para não atrasar muito o treinamento
            time.sleep(dt)


    env.close()
    cv2.destroyAllWindows()

    # Salvar Tabela Q final ao sair
    if q_table: # Verifica se a tabela não está vazia
        print("Salvando Tabela Q final...")
        try:
            with open(Q_TABLE_FILENAME, "wb") as f:
                pickle.dump(q_table, f)
            print(f"Tabela Q salva em {Q_TABLE_FILENAME}")
        except Exception as e:
            print(f"Erro ao salvar Tabela Q final: {e}")

if __name__ == "__main__":
    main()
