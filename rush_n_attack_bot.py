import retro
import cv2
import numpy as np
import time
import os # Importar os para verificar a existência de arquivos

# Intervalo de tempo entre frames (aproximado)
dt = 1/60.0

# Lista global para armazenar os rastreadores de inimigos
enemy_trackers = [] # Será reiniciada ao carregar um estado
next_tracker_id = 0 # Será reiniciado ao carregar um estado

# Constantes para tipos de inimigos
ENEMY_TYPE_1 = 1
ENEMY_TYPE_2 = 2
ENEMY_TYPE_3 = 3
ENEMY_TYPE_MINE = 4

# Nome do arquivo para o save state
SAVE_STATE_FILENAME = "rushnattack_mines_state.sav"
LOAD_STATE_AT_STARTUP = True # Mude para False se não quiser carregar automaticamente

class EnemyTracker:
    def __init__(self, initial_measurement, enemy_type, tracker_id):
        self.id = tracker_id
        self.type = enemy_type
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.diag([0.25, 0.25, 1.0, 1.0]).astype(np.float32) # Q
        self.kf.measurementNoiseCov = np.diag([25.0, 25.0]).astype(np.float32)     # R

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

# --- FUNÇÕES DE DETECÇÃO ---
def detect_player(frame):
    lower_blue = np.array([110, 200, 150])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(frame, lower_blue, upper_blue)
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=10)
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
    return (0, 0)

def detect_enemy1(frame, block_height=20, block_width=60):
    height, width, _ = frame.shape
    white = np.array([240, 240, 240], dtype=np.uint8)
    min_white = 150
    positions = []
    y = 570
    x = 0
    while x < width:
        if x + block_width > width:
            break
        block_frame = frame[y:y+block_height, x:x+block_width]
        mask_white = np.all(block_frame >= white, axis=2)
        sum_white = np.sum(mask_white)
        if sum_white > min_white:
            positions.append((x+30, y-10, ENEMY_TYPE_1))
            x += 60
        else:
            x += 10
    return positions

def detect_enemy2(frame):
    lower_orange = np.array([11, 255, 216])
    upper_orange = np.array([7, 255, 255])
    mask = cv2.inRange(frame, lower_orange, upper_orange)

    # Dilatação
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=10)
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
                centers.append((cx, cy, 2))
    return centers


def detect_enemy3(frame):
    lower_yellow = np.array([15, 197, 244])
    upper_yellow = np.array([15, 199, 248])
    mask = cv2.inRange(frame, lower_yellow, upper_yellow)
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.erode(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    min_area = 3
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
        # ESTES PARÂMETROS AGORA DEVEM DESCREVER A COR DA MINA
        # Exemplo: Se a mina for laranja/marrom (BGR: 59, 165, 228)
        target_color_params = {
            'lower_bgr': np.array([22, 20, 20]), # BGR Limite inferior da COR DA MINA
            'upper_bgr': np.array([100, 200, 255])  # BGR Limite superior da COR DA MINA
        }
    if mine_filter_params is None:
        mine_filter_params = {
            'min_area': 10,
            'max_area': 40,
            'morph_open_ksize': 3,
            'morph_open_iter': 1,
            'morph_close_ksize': 3,
            'morph_close_iter': 1
        }

    roi_y_start = max(0, scan_center_y - y_margin)
    roi_y_end = min(frame_height, scan_center_y + y_margin)

    if roi_y_start >= roi_y_end: return []
    roi = frame_bgr[roi_y_start:roi_y_end, :]
    if roi.size == 0: return []

    # 1. Criar máscara para pixels que correspondem à COR DA MINA
    # Não chamaremos mais de 'white_mask', mas 'target_mask'
    target_mask = cv2.inRange(roi, target_color_params['lower_bgr'], target_color_params['upper_bgr'])

    # 2. Operações Morfológicas para limpar a target_mask
    processed_mask = target_mask # Começa com a target_mask
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

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                full_frame_cx = cx
                full_frame_cy = cy + roi_y_start
                detected_mines_list.append((full_frame_cx, full_frame_cy, ENEMY_TYPE_MINE))
                cv2.drawContours(roi, [cnt], -1, (0, 255, 255), 1) # Amarelo para esta versão
    return detected_mines_list

# --- FIM DAS FUNÇÕES DE DETECÇÃO ---

def closest_enemy(player_pos, enemy_predictions):
    best_idx, best_dist = -1, float('inf')
    px, py = player_pos
    if not enemy_predictions: return best_idx, best_dist
    for i, (ex, ey, _) in enumerate(enemy_predictions):
        d = np.hypot(px-ex, py-ey)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx, best_dist

def main():
    global enemy_trackers, next_tracker_id, obs # obs precisa ser global se modificado por load_state_action

    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()

    b_press_frames = 0
    a_press_frames = 0

    press_down = False

    max_misses = 3
    association_threshold = 75
    mine_jump_distance = 120

    # Função auxiliar para resetar o estado do agente
    def reset_agent_state():
        global enemy_trackers, next_tracker_id
        enemy_trackers = []
        next_tracker_id = 0
        # b_press_frames e press_down serão resetados/recalculados no loop
        print("Agent state reset.")

    # Tenta carregar o estado no início
    if LOAD_STATE_AT_STARTUP and os.path.exists(SAVE_STATE_FILENAME):
        print(f"Tentando carregar estado de {SAVE_STATE_FILENAME}...")
        try:
            with open(SAVE_STATE_FILENAME, "rb") as f:
                state_data = f.read()
            if env.em.set_state(state_data): # Usa o método do emulador
                print("Estado do emulador carregado com sucesso.")
                obs, _, _, _ = env.step(np.zeros(env.action_space.shape[0], dtype=np.uint8)) # Passo dummy para atualizar obs
                reset_agent_state() # Reseta o estado do agente
            else:
                print("Falha ao carregar o estado do emulador. Começando do início.")
                obs = env.reset() # Garante um reinício limpo
        except Exception as e:
            print(f"Erro ao carregar estado: {e}. Começando do início.")
            obs = env.reset()
    elif LOAD_STATE_AT_STARTUP:
        print(f"Arquivo de save state '{SAVE_STATE_FILENAME}' não encontrado. Começando do início.")

    # Inicializa/reseta o estado do agente se não foi carregado um save state que já o fez
    if not (LOAD_STATE_AT_STARTUP and os.path.exists(SAVE_STATE_FILENAME)): # Evita reset duplo
        reset_agent_state() # Garante que o estado do agente está limpo no começo normal

    while True:
        obs_resized = cv2.resize(obs, (960, 900))[200:900-50, :]
        frame_rgb = obs_resized.copy()
        frame_hsv = cv2.cvtColor(obs_resized, cv2.COLOR_RGB2HSV)

        player_pos = detect_player(frame_hsv)

        for tracker in enemy_trackers:
            tracker.predict()

        detections_e1 = detect_enemy1(frame_rgb)
        detections_e2 = detect_enemy2(frame_hsv)
        detections_e3 = detect_enemy3(frame_hsv)
        detections_mines = detect_landmine(frame_rgb)

        all_current_detections = []
        for x_coord, y_coord, type_val in detections_e1: all_current_detections.append({'pos': (x_coord,y_coord), 'type': type_val, 'matched': False})
        for x_coord, y_coord, type_val in detections_e2: all_current_detections.append({'pos': (x_coord,y_coord), 'type': type_val, 'matched': False})
        for x_coord, y_coord, type_val in detections_e3: all_current_detections.append({'pos': (x_coord,y_coord), 'type': type_val, 'matched': False})
        for x_coord, y_coord, type_val in detections_mines: all_current_detections.append({'pos': (x_coord,y_coord), 'type': type_val, 'matched': False})

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

        enemies_for_logic = []
        for tracker in enemy_trackers:
            if tracker.misses < max_misses and tracker.age > 1:
                 enemies_for_logic.append((tracker.predicted_pos[0], tracker.predicted_pos[1], tracker.type))

        action = np.zeros(env.action_space.shape[0], dtype=np.uint8)
        current_frame_press_down = False
        acted_on_enemy_this_frame = False # Flag para controlar se uma ação específica foi tomada

        if enemies_for_logic:
            idx, dist = closest_enemy(player_pos, enemies_for_logic)
            if idx != -1:
                ex, ey, enemy_type = enemies_for_logic[idx]

                # --- Lógica específica para MINAS ---
                if enemy_type == ENEMY_TYPE_MINE:
                    if dist < mine_jump_distance:  # Mina perto para pular
                        # Define a intenção de abaixar para ESTE frame se a condição for atendida
                        current_frame_press_down = True

                        # INICIA o timer de tiro APENAS SE NÃO ESTIVER ATIVO
                        if not (a_press_frames > 0): # Verifica se a_press_frames já está contando
                            print(f"INFO: Mina PRÓXIMA detectada. Dist: {dist:.2f}. INICIANDO tiro com botão 8.")
                            a_press_frames = 5 # Duração do tiro (ex: 5 frames)
                        # Se a_press_frames já estiver > 0, significa que estamos no meio de um tiro
                        # e current_frame_press_down (se True) manterá o agachamento.

                        acted_on_enemy_this_frame = True
                        # Nenhuma ação de movimento aqui, para o jogador ficar parado
                    elif dist < 150:  # Mina um pouco mais longe, perseguir
                        # Certifique-se de que não estamos no meio de um tiro em uma mina anterior
                        if not (a_press_frames > 0): # Só persegue se não estiver atirando
                            print(f"INFO: Mina detectada (longe para pulo). Dist: {dist:.2f}. Perseguindo.")
                            action[7 if player_pos[0] < ex else 6] = 1
                            acted_on_enemy_this_frame = True
                # --- Lógica para OUTROS tipos de inimigos (APENAS SE NÃO AGIMOS NA MINA) ---
                if not acted_on_enemy_this_frame:
                    if enemy_type == ENEMY_TYPE_3: # Exemplo: Inimigo que requer abaixar
                        if dist < 100: # Distância de ataque para inimigo tipo 3
                            print(f"INFO: Inimigo Tipo 3 PRÓXIMO. Dist: {dist:.2f}. Abaixando e preparando tiro.")
                            current_frame_press_down = True
                            b_press_frames = 3 # Ajuste conforme necessário
                            action[7 if player_pos[0] < ex else 6] = 1 # Mover enquanto ataca
                            acted_on_enemy_this_frame = True
                        elif dist < 150: # Perseguir inimigo tipo 3
                            print(f"INFO: Inimigo Tipo 3 detectado. Dist: {dist:.2f}. Perseguindo.")
                            action[7 if player_pos[0] < ex else 6] = 1
                            acted_on_enemy_this_frame = True
                    # Adicione aqui elif para ENEMY_TYPE_1, ENEMY_TYPE_2 se tiverem lógicas especiais
                    # Senão, uma lógica geral de ataque/perseguição:
                    elif dist < 100: # Distância geral de ataque para outros inimigos
                        print(f"INFO: Inimigo {enemy_type} PRÓXIMO. Dist: {dist:.2f}. Preparando tiro.")
                        # current_frame_press_down permanece False por padrão para ataque em pé
                        b_press_frames = 3 # Ajuste conforme necessário
                        action[7 if player_pos[0] < ex else 6] = 1 # Mover enquanto ataca
                        acted_on_enemy_this_frame = True
                    elif dist < 250 and enemy_type != ENEMY_TYPE_MINE: # Distância geral de perseguição
                        print(f"INFO: Inimigo {enemy_type} detectado. Dist: {dist:.2f}. Perseguindo.")
                        action[7 if player_pos[0] < ex else 6] = 1
                        acted_on_enemy_this_frame = True

                # Se, mesmo após checar todos os tipos, o inimigo mais próximo ainda estiver muito longe
                if not acted_on_enemy_this_frame:
                    print(f"INFO: Inimigo {enemy_type} muito longe (Dist: {dist:.2f}). Movendo por padrão.")
                    action[7] = 1 # Mover para a direita por padrão

            else: # Nenhum inimigo rastreável mais próximo (idx == -1)
                print("INFO: Nenhum inimigo rastreável próximo. Movendo por padrão.")
                action[7] = 1
        else: # Lista `enemies_for_logic` vazia
            print("INFO: Nenhum inimigo na lógica. Movendo por padrão.")
            action[7] = 1

        if b_press_frames > 0:
            print(f"ACTION: ATACANDO! b_frames_restantes: {b_press_frames}, Abaixado: {current_frame_press_down}")
            attack_action_this_frame = np.zeros(env.action_space.shape[0], dtype=np.uint8)
            attack_action_this_frame[0] = 1 # Botão 'B' (ATAQUE)
            if current_frame_press_down:
                attack_action_this_frame[5] = 1 # Botão 'DOWN' (ABAIXAR)
            action = attack_action_this_frame
            b_press_frames -= 1

        if a_press_frames > 0:
            jump_action_final = np.zeros(env.action_space.shape[0], dtype=np.uint8)
            jump_action_final[4] = 1  # Pressiona 'A' (PULO)
            jump_action_final[7] = 1     # Mantém movimento para frente durante o pulo
            action = jump_action_final # Pulo sobrepõe outras ações
            a_press_frames -= 1

        elif not acted_on_enemy_this_frame and np.count_nonzero(action) == 0 : # Garante que se nenhuma ação foi definida, ele se move
                print("ACTION: Nenhuma ação de inimigo ou ataque, movendo por padrão.")
                action[7] = 1 # Mover para a direita se nenhuma outra ação foi tomada

        press_down = current_frame_press_down
        # ... (resto do loop: env.step(action), visualização, etc.) ...
        # obs é atualizado aqui para o próximo frame
        next_obs, _, done, _ = env.step(action) # Renomeado para next_obs para clareza
        debug_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        px, py = player_pos
        cv2.rectangle(debug_frame, (px-30, py-60), (px+30, py+60), (255, 0, 0), 2)

        for tracker in enemy_trackers:
            pred_x, pred_y = tracker.predicted_pos
            color = (0,0,0)
            if tracker.type == ENEMY_TYPE_1: color = (0, 0, 255)
            elif tracker.type == ENEMY_TYPE_2: color = (0, 255, 0)
            elif tracker.type == ENEMY_TYPE_3: color = (0, 255, 255)
            elif tracker.type == ENEMY_TYPE_MINE: color = (255, 0, 255)
            cv2.circle(debug_frame, (pred_x, pred_y), 8, color, -1)
            cv2.putText(debug_frame, f"T{tracker.type}", (pred_x + 10, pred_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("Rush'n Attack - Kalman Prediction", debug_frame)

        # Processamento de Teclas para Save/Load/Quit
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27: # ESC
            break
        elif key_pressed == ord('s'): # Tecla 'S' para Salvar Estado
            print("Salvando estado atual do emulador...")
            try:
                state_data = env.em.get_state()
                with open(SAVE_STATE_FILENAME, "wb") as f:
                    f.write(state_data)
                print(f"Estado salvo em {SAVE_STATE_FILENAME}")
            except Exception as e:
                print(f"Erro ao salvar estado: {e}")
        elif key_pressed == ord('l'): # Tecla 'L' para Carregar Estado
            if os.path.exists(SAVE_STATE_FILENAME):
                print(f"Carregando estado de {SAVE_STATE_FILENAME}...")
                try:
                    with open(SAVE_STATE_FILENAME, "rb") as f:
                        state_data = f.read()
                    if env.em.set_state(state_data):
                        print("Estado do emulador carregado com sucesso.")
                        next_obs, _, done, _ = env.step(np.zeros(env.action_space.shape[0], dtype=np.uint8)) # Passo dummy
                        reset_agent_state() # Reseta o estado do agente
                        # As variáveis de controle de ataque também devem ser resetadas
                        b_press_frames = 0
                        press_down = False # ou recalculado com base no novo 'obs'
                    else:
                        print("Falha ao carregar o estado do emulador.")
                except Exception as e:
                    print(f"Erro ao carregar estado: {e}")
            else:
                print(f"Arquivo de save state '{SAVE_STATE_FILENAME}' não encontrado.")

        obs = next_obs # Atualiza obs para a próxima iteração

        if done:
            print("Fim do jogo (done=True). Reiniciando episódio...")
            obs = env.reset()
            reset_agent_state() # Reseta o estado do agente
            b_press_frames = 0
            press_down = False

        time.sleep(dt)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
