import retro
import cv2
import numpy as np
import time
import os
import pickle # Para salvar/carregar a tabela Q

# --- Configurações de Treinamento ---
VISUALIZE = True  # Mude para False para treinamento headless (muito mais rápido)
# Se VISUALIZE = False, dt e time.sleep não terão efeito prático na velocidade do loop principal.
dt = 0.0 # Para treinamento rápido. Mantenha > 0 (ex: 1/60.0) se quiser ver em velocidade normal com VISUALIZE = True.

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
LOAD_EMULATOR_STATE_AT_STARTUP = False # Carregar estado do *emulador*

# --- Parâmetros do Q-learning ---
ALPHA = 0.1  # Taxa de aprendizado
GAMMA = 0.99 # Fator de desconto
EPSILON = 1.0 # Taxa de exploração inicial
EPSILON_DECAY = 0.9995 # Taxa de decaimento do epsilon por episódio
MIN_EPSILON = 0.05 # Epsilon mínimo
q_table = {}

Q_TABLE_FILENAME = "rushnattack_q_table_v2.pkl" # Nome do arquivo da tabela Q
LOAD_Q_TABLE_AT_STARTUP = True # Carregar tabela Q aprendida
SAVE_Q_TABLE_PERIODICALLY = True
SAVE_Q_TABLE_EPISODE_INTERVAL = 10 # Salvar a cada X episódios

# --- Discretização do Espaço de Estados ---
PLAYER_X_BINS = np.linspace(0, 960, 15)
ENEMY_DX_BINS = np.linspace(-250, 250, 10)
ENEMY_DY_BINS = np.linspace(-150, 150, 8)
ENEMY_TYPE_CATEGORIES = sorted([ENEMY_TYPE_1, ENEMY_TYPE_2, ENEMY_TYPE_3, ENEMY_TYPE_MINE])
ENEMY_TYPE_BINS = np.append(np.array(ENEMY_TYPE_CATEGORIES) - 0.5, max(ENEMY_TYPE_CATEGORIES) + 0.5)


# --- Definição do Espaço de Ações Discretas ---
# Rush'n Attack buttons: ['B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'FIRE', 'SPECIAL', 'UNUSED1', 'UNUSED2']
# B: 0, A: 7, FIRE: 8
# UP: 3, DOWN: 4, LEFT: 5, RIGHT: 6
# 0: Direita, 1: Esquerda, 2: Atacar (B), 3: Pular (A),
# 4: Direita+Atacar(B), 5: Direita+Pular(A), 6: Baixo, 7: Baixo+Atacar(B),
# 8: Atirar (FIRE), 9: No-Op
NUM_DISCRETE_ACTIONS = 10
WASTEFUL_ACTION_INDICES = {4,5,8,6,0} # Ações que podem ser desperdiçadas

def get_env_action(discrete_action_index, num_buttons):
    env_action = np.zeros(num_buttons, dtype=np.uint8)
    if discrete_action_index == 0: # Direita
        env_action[7] = 1 # RIGHT_BUTTON_INDEX
    elif discrete_action_index == 1: # Esquerda
        env_action[6] = 1 # LEFT_BUTTON_INDEX
    elif discrete_action_index == 2: # Atacar (B)
        env_action[0] = 1 # B_BUTTON_INDEX
    elif discrete_action_index == 3: # Pular (A)
        env_action[4] = 1 # A_BUTTON_INDEX
    elif discrete_action_index == 4: # Direita + Atacar (B)
        env_action[6] = 1
        env_action[0] = 1
    elif discrete_action_index == 5: # Direita + Pular (A)
        env_action[6] = 1
        env_action[7] = 1
    elif discrete_action_index == 6: # Baixo
        env_action[5] = 1 # DOWN_BUTTON_INDEX (original era 5, mas no array de botões é 4)
    elif discrete_action_index == 7: # Baixo + Atacar (B)
        env_action[5] = 1
        env_action[0] = 1
    elif discrete_action_index == 8: # Atirar (FIRE)
        env_action[8] = 1 # FIRE_BUTTON_INDEX
    # Ação 9: No-Op (não faz nada) - env_action já é zeros
    return env_action

class EnemyTracker:
    def __init__(self, initial_measurement, enemy_type, tracker_id):
        self.id = tracker_id
        self.type = enemy_type
        self.kf = cv2.KalmanFilter(4, 2)
        actual_dt = dt if dt > 0 else 1/60.0 # Kalman Filter precisa de um dt > 0
        self.kf.transitionMatrix = np.array([[1,0,actual_dt,0],[0,1,0,actual_dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.diag([0.25, 0.25, 1.0, 1.0]).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([25.0, 25.0]).astype(np.float32)
        initial_x = float(initial_measurement[0])
        initial_y = float(initial_measurement[1])
        self.kf.statePost = np.array([[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.array([[25.0,0,0,0],[0,25.0,0,0],[0,0,1000.0,0],[0,0,0,1000.0]],np.float32)
        self.predicted_pos = (int(self.kf.statePost[0,0]), int(self.kf.statePost[1,0]))
        self.hits = 1
        self.misses = 0
        self.age = 0
    def predict(self):
        prediction = self.kf.predict()
        self.predicted_pos = (int(prediction[0,0]), int(prediction[1,0]))
        self.age += 1
        return self.predicted_pos
    def update(self, measurement):
        measurement_vec = np.array([measurement[0],measurement[1]],dtype=np.float32).reshape(2,1)
        self.kf.correct(measurement_vec)
        self.hits += 1; self.misses = 0
        self.predicted_pos = (int(self.kf.statePost[0,0]), int(self.kf.statePost[1,0]))
    def increment_misses(self): self.misses += 1

# --- FUNÇÕES DE DETECÇÃO (sem grandes alterações na lógica interna) ---
def detect_player(frame_hsv):
    lower_blue = np.array([110, 200, 150])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)
    kernel = np.ones((10,10), np.uint8) # Pode testar iterations=5 aqui
    dilated_mask = cv2.dilate(mask, kernel, iterations=5)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    min_area = 400 # Ajustado
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0: centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
    return centers[0] if centers else (0,0)

def detect_enemy1(frame_rgb): # Mantido como estava
    height, width, _ = frame_rgb.shape; white = np.array([240,240,240],dtype=np.uint8); min_white = 150
    positions = []; y = 570; x = 0; block_height=20; block_width=60
    while x < width:
        if x+block_width > width: break
        block_frame = frame_rgb[y:y+block_height, x:x+block_width]
        if block_frame.shape[0]<block_height or block_frame.shape[1]<block_width: break
        if np.sum(np.all(block_frame >= white, axis=2)) > min_white:
            positions.append((x+30,y-10,ENEMY_TYPE_1)); x += 60
        else: x += 10
    return positions

def detect_enemy2(frame_hsv): # Mantido como estava
    frame_resized = frame_hsv[570:590,:]; lower_orange=np.array([5,250,200]); upper_orange=np.array([7,255,255])
    mask = cv2.inRange(frame_resized,lower_orange,upper_orange)
    dilated_mask = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    centers = []; min_area = 130
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            if M['m00']!=0: centers.append((int(M['m10']/M['m00']),560,ENEMY_TYPE_2))
    return centers

def detect_enemy3(frame_hsv): # Mantido como estava
    lower_yellow=np.array([15,197,244]); upper_yellow=np.array([15,199,248])
    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    dilated_mask = cv2.erode(mask, np.ones((10,10),np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    centers = []; min_area = 3
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            if M['m00']!=0: centers.append((int(M['m10']/M['m00']),int(M['m01']/M['m00']),ENEMY_TYPE_3))
    return centers

def detect_landmine(frame_bgr): # Mantido como estava, parâmetros default ajustados antes
    target_color_params={'lower_bgr':np.array([20,20,20]),'upper_bgr':np.array([100,100,100])}
    mine_filter_params={'min_area':8,'max_area':50,'morph_open_ksize':3,'morph_open_iter':1,'morph_close_ksize':3,'morph_close_iter':1}
    scan_center_y=645; y_margin=5; frame_height,_=frame_bgr.shape[:2]
    roi_y_start=max(0,scan_center_y-y_margin); roi_y_end=min(frame_height,scan_center_y+y_margin)
    if roi_y_start>=roi_y_end: return []
    roi=frame_bgr[roi_y_start:roi_y_end,:];
    if roi.size==0: return []
    target_mask=cv2.inRange(roi,target_color_params['lower_bgr'],target_color_params['upper_bgr'])
    processed_mask=target_mask # ... (operações morfológicas omitidas por brevidade, mas estão no original)
    open_ksize=mine_filter_params.get('morph_open_ksize',0); open_iter=mine_filter_params.get('morph_open_iter',0)
    if open_ksize>0 and open_iter>0: processed_mask=cv2.morphologyEx(processed_mask,cv2.MORPH_OPEN,np.ones((open_ksize,open_ksize),np.uint8),iterations=open_iter)
    close_ksize=mine_filter_params.get('morph_close_ksize',0); close_iter=mine_filter_params.get('morph_close_iter',0)
    if close_ksize>0 and close_iter>0: processed_mask=cv2.morphologyEx(processed_mask,cv2.MORPH_CLOSE,np.ones((close_ksize,close_ksize),np.uint8),iterations=close_iter)
    contours,_=cv2.findContours(processed_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    detected_mines_list=[]; min_a=mine_filter_params['min_area']; max_a=mine_filter_params['max_area']
    for cnt in contours:
        if min_a<=cv2.contourArea(cnt)<=max_a:
            M=cv2.moments(cnt)
            if M["m00"]!=0: detected_mines_list.append((int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])+roi_y_start,ENEMY_TYPE_MINE))
    return detected_mines_list

# --- FIM DAS FUNÇÕES DE DETECÇÃO ---

def discretize_value(value, bins): return np.digitize(value, bins) - 1

def get_q_state(player_pos, enemies_for_logic):
    player_x_discrete = discretize_value(player_pos[0], PLAYER_X_BINS)
    closest_enemy_dx_discrete = discretize_value(0, ENEMY_DX_BINS)
    closest_enemy_dy_discrete = discretize_value(0, ENEMY_DY_BINS)
    closest_enemy_type_discrete = len(ENEMY_TYPE_BINS) - 1 # Default: no significant enemy

    if enemies_for_logic:
        px, py = player_pos; best_dist = float('inf'); closest_enemy_data = None
        for ex, ey, enemy_type in enemies_for_logic:
            d = np.hypot(px-ex, py-ey)
            if d < best_dist: best_dist=d; closest_enemy_data=(ex,ey,enemy_type)
        if closest_enemy_data and best_dist < 250: # Consider enemy if reasonably close
            ex, ey, enemy_type = closest_enemy_data
            closest_enemy_dx_discrete = discretize_value(ex-px, ENEMY_DX_BINS)
            closest_enemy_dy_discrete = discretize_value(ey-py, ENEMY_DY_BINS)
            type_bin_index = discretize_value(enemy_type, ENEMY_TYPE_BINS) # np.digitize is 1-indexed
            if 0 <= type_bin_index < len(ENEMY_TYPE_BINS)-1 : closest_enemy_type_discrete = type_bin_index
    return (player_x_discrete, closest_enemy_dx_discrete, closest_enemy_dy_discrete, closest_enemy_type_discrete)

def calculate_reward_v2(player_x_curr, player_x_prev, score_curr, score_prev, lives_curr, lives_prev,
                        num_enemies_curr, num_enemies_prev, action_idx,
                        done_flag, game_won_flag=False):
    reward = 0.0
    reward -= 0.02  # Penalidade por frame para incentivar rapidez

    score_delta = score_curr - score_prev
    reward += score_delta * 1.0  # Recompensa principal por pontuação

    progress_delta = player_x_curr - player_x_prev
    if progress_delta > 1: reward += progress_delta * 0.02
    elif progress_delta < -5: reward -= 0.1 # Penalidade por recuar

    if lives_curr < lives_prev: reward -= 150.0

    enemies_killed = num_enemies_prev - num_enemies_curr # Inimigos visíveis
    if enemies_killed > 0 : reward += enemies_killed * 15.0 # Recompensa por remover inimigos da tela

    # Penalidade por ações desnecessárias (se não houver inimigos *próximos*)
    # O estado já considera "sem inimigos próximos". Se num_enemies_prev == 0, significa que nenhum inimigo foi rastreado.
    if num_enemies_prev == 0 and action_idx in WASTEFUL_ACTION_INDICES:
        reward -= 2.0 # Penalidade menor

    if action_idx == 7:
        reward += 15.0

    if done_flag:
        if game_won_flag: reward += 500.0
        elif lives_curr == 0: reward -= 200.0
    return reward

def process_frame_and_get_detections(obs_frame_full_color):
    global enemy_trackers, next_tracker_id
    obs_resized = cv2.resize(obs_frame_full_color, (960,900))[200:900-50, :]
    frame_rgb = obs_resized.copy()
    frame_hsv = cv2.cvtColor(obs_resized, cv2.COLOR_RGB2HSV)

    current_player_pos = detect_player(frame_hsv)
    if current_player_pos == (0,0) and hasattr(process_frame_and_get_detections, 'last_known_player_pos'):
        current_player_pos = process_frame_and_get_detections.last_known_player_pos
    else:
        process_frame_and_get_detections.last_known_player_pos = current_player_pos

    for tracker in enemy_trackers: tracker.predict()
    detections = detect_enemy1(frame_rgb) + detect_enemy2(frame_hsv) + \
                 detect_enemy3(frame_hsv) + detect_landmine(frame_rgb) # Passar frame_rgb para detect_landmine

    all_current_detections = [{'pos':(x,y),'type':t,'matched':False} for x,y,t in detections]
    max_misses=3; association_threshold=75; active_trackers_this_frame=[]
    for tracker in enemy_trackers:
        best_match_idx=-1; min_dist=association_threshold
        for i,det in enumerate(all_current_detections):
            if not det['matched'] and det['type']==tracker.type:
                dist=np.hypot(tracker.predicted_pos[0]-det['pos'][0], tracker.predicted_pos[1]-det['pos'][1])
                if dist<min_dist: min_dist=dist; best_match_idx=i
        if best_match_idx!=-1:
            tracker.update(all_current_detections[best_match_idx]['pos'])
            all_current_detections[best_match_idx]['matched']=True
            active_trackers_this_frame.append(tracker)
        else:
            tracker.increment_misses()
            if tracker.misses<=max_misses: active_trackers_this_frame.append(tracker)
    enemy_trackers=active_trackers_this_frame
    for det in all_current_detections:
        if not det['matched']:
            enemy_trackers.append(EnemyTracker(det['pos'],det['type'],next_tracker_id)); next_tracker_id+=1

    current_enemies_for_logic = [(tr.predicted_pos[0],tr.predicted_pos[1],tr.type) \
                                 for tr in enemy_trackers if tr.misses<max_misses and tr.age>0]
    return current_player_pos, current_enemies_for_logic, len(current_enemies_for_logic), frame_rgb

def main():
    global enemy_trackers, next_tracker_id, obs, q_table, EPSILON

    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset() # obs inicial (frame completo)
    num_buttons = env.action_space.shape[0]

    if LOAD_Q_TABLE_AT_STARTUP and os.path.exists(Q_TABLE_FILENAME):
        print(f"Carregando Tabela Q de {Q_TABLE_FILENAME}...")
        try:
            with open(Q_TABLE_FILENAME,"rb") as f: q_table=pickle.load(f)
            print("Tabela Q carregada.")
        except Exception as e: print(f"Erro ao carregar Tabela Q: {e}. Nova tabela."); q_table={}
    else: q_table={}; print("Nova Tabela Q iniciada.")

    # Variáveis para cálculo de recompensa e estatísticas
    prev_player_x, prev_score, prev_lives, prev_num_enemies = 0,0,3,0
    total_episode_reward = 0.0
    episode_count = 0

    def reset_episode_vars(initial_obs_frame, initial_info):
        nonlocal prev_player_x, prev_score, prev_lives, prev_num_enemies, total_episode_reward
        # A posição inicial real e inimigos serão definidos no primeiro process_frame do loop
        p_pos, _, num_enemies, _ = process_frame_and_get_detections(initial_obs_frame)
        prev_player_x = p_pos[0] if p_pos[0] != 0 else 0 # Evita ficar preso em 0 se a detecção inicial falhar
        prev_score = initial_info.get('score',0)
        prev_lives = initial_info.get('lives',3)
        prev_num_enemies = num_enemies
        total_episode_reward = 0.0

        global enemy_trackers, next_tracker_id # Reiniciar rastreadores de inimigos
        enemy_trackers = []
        next_tracker_id = 0
        process_frame_and_get_detections.last_known_player_pos = (0,0) # Reset last known pos

    _,_,_,initial_info_dummy = env.step(np.zeros(num_buttons,dtype=np.uint8)) # Pegar info inicial
    reset_episode_vars(obs, initial_info_dummy)

    if LOAD_EMULATOR_STATE_AT_STARTUP and os.path.exists(SAVE_STATE_FILENAME):
        print(f"Carregando estado do emulador de {SAVE_STATE_FILENAME}...")
        # ... (lógica de carregar estado do emulador, similar à anterior)
        # Se carregar, chamar reset_episode_vars(obs, info_do_estado_carregado)

    running = True
    while running:
        player_pos_s, enemies_s_logic, num_enemies_s, frame_rgb_s = process_frame_and_get_detections(obs)
        current_q_s = get_q_state(player_pos_s, enemies_s_logic)

        if np.random.rand() < EPSILON: action_idx = np.random.randint(0,NUM_DISCRETE_ACTIONS)
        else:
            q_vals = q_table.get(current_q_s, np.zeros(NUM_DISCRETE_ACTIONS))
            action_idx = np.argmax(q_vals) if not np.all(q_vals==0) else np.random.randint(0,NUM_DISCRETE_ACTIONS)

        env_action = get_env_action(action_idx, num_buttons)
        next_obs, _, done, info = env.step(env_action) # next_obs é o frame completo

        player_pos_s_prime, enemies_s_prime_logic, num_enemies_s_prime, _ = process_frame_and_get_detections(next_obs)
        next_q_s_prime = get_q_state(player_pos_s_prime, enemies_s_prime_logic)

        score_curr = info.get('score', prev_score)
        lives_curr = info.get('lives', prev_lives)
        game_won = False # Placeholder para lógica de vitória


        custom_reward = calculate_reward_v2(
            player_pos_s[0], prev_player_x, score_curr, prev_score, lives_curr, prev_lives,
            num_enemies_s_prime, num_enemies_s, # num_enemies_s é o prev_num_enemies para esta transição
            action_idx, done, game_won
        )
        total_episode_reward += custom_reward

        old_q = q_table.get(current_q_s,np.zeros(NUM_DISCRETE_ACTIONS))[action_idx]
        next_max_q = 0.0 if done else np.max(q_table.get(next_q_s_prime,np.zeros(NUM_DISCRETE_ACTIONS)))
        new_q = old_q + ALPHA * (custom_reward + GAMMA * next_max_q - old_q)
        if current_q_s not in q_table: q_table[current_q_s] = np.zeros(NUM_DISCRETE_ACTIONS)
        q_table[current_q_s][action_idx] = new_q

        obs = next_obs
        prev_player_x = player_pos_s[0]
        prev_score = score_curr
        prev_lives = lives_curr
        prev_num_enemies = num_enemies_s_prime # O número de inimigos no estado s' se torna o "prev" para o próximo s

        if VISUALIZE:
            debug_bgr = cv2.cvtColor(frame_rgb_s, cv2.COLOR_RGB2BGR)
            cv2.rectangle(debug_bgr,(player_pos_s[0]-15,player_pos_s[1]-30),(player_pos_s[0]+15,player_pos_s[1]+30),(255,0,0),1)
            for tr in enemy_trackers: # Desenhar todos os trackers ativos
                pred_x,pred_y=tr.predicted_pos; color=(0,0,0)
                if tr.type==ENEMY_TYPE_1:color=(0,0,255)
                elif tr.type==ENEMY_TYPE_2:color=(0,255,0)
                elif tr.type==ENEMY_TYPE_3:color=(0,255,255)
                elif tr.type==ENEMY_TYPE_MINE:color=(255,0,255)
                cv2.circle(debug_bgr,(pred_x,pred_y),5,color,-1)
            cv2.putText(debug_bgr,f"Ep:{episode_count} Rew:{total_episode_reward:.1f} Eps:{EPSILON:.3f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
            cv2.putText(debug_bgr,f"L:{lives_curr} S:{score_curr} Act:{action_idx} R:{custom_reward:.2f}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
            cv2.putText(debug_bgr,f"Enemies S:{num_enemies_s} S':{num_enemies_s_prime}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)


            cv2.imshow("Rush'n Attack - Q-learning", debug_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: running = False; break
            # Adicionar 's' para salvar Q-table manualmente e 'p' para pausar/alterar epsilon, etc.
            elif key == ord('p'): # Pausa para inspeção
                print("Pausado. Pressione qualquer tecla no console para continuar...")
                input()

        if done:
            print(f"Episódio {episode_count} Recompensa: {total_episode_reward:.2f} Epsilon: {EPSILON:.4f} Q-Size: {len(q_table)}")
            _,_,_,info_reset = env.step(np.zeros(num_buttons,dtype=np.uint8)) # Para info
            obs = env.reset()
            reset_episode_vars(obs, info_reset)
            episode_count += 1
            if EPSILON > MIN_EPSILON: EPSILON *= EPSILON_DECAY
            if SAVE_Q_TABLE_PERIODICALLY and episode_count % SAVE_Q_TABLE_EPISODE_INTERVAL == 0:
                try:
                    with open(Q_TABLE_FILENAME,"wb") as f: pickle.dump(q_table,f)
                    print(f"Tabela Q salva em {Q_TABLE_FILENAME}")
                except Exception as e: print(f"Erro ao salvar Tabela Q: {e}")

        if VISUALIZE and dt > 0.0001: time.sleep(dt)

    env.close()
    if VISUALIZE: cv2.destroyAllWindows()
    if q_table:
        print("Salvando Tabela Q final...")
        try:
            with open(Q_TABLE_FILENAME,"wb") as f: pickle.dump(q_table,f)
            print(f"Tabela Q salva em {Q_TABLE_FILENAME}")
        except Exception as e: print(f"Erro ao salvar Tabela Q: {e}")

if __name__ == "__main__":
    main()
