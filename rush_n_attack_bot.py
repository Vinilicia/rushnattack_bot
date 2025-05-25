import retro
import cv2
import numpy as np
import time

# Intervalo de tempo entre frames (aproximado)
dt = 1/60.0

# Lista global para armazenar os rastreadores de inimigos
enemy_trackers = []
next_tracker_id = 0

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

        # Inicializa statePost DIRETAMENTE com a primeira medição e velocidade zero
        self.kf.statePost = np.array([[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32)

        # Define a covariância do erro inicial (P0):
        # A incerteza da posição inicial é igual à incerteza da medição (R).
        # A incerteza da velocidade inicial é alta.
        self.kf.errorCovPost = np.array([[25.0, 0.0,  0.0,    0.0], # Var(x) = R_x
                                         [0.0, 25.0,  0.0,    0.0], # Var(y) = R_y
                                         [0.0,  0.0, 1000.0,  0.0], # Var(vx) alta
                                         [0.0,  0.0,  0.0, 1000.0]],np.float32) # Var(vy) alta

        self.predicted_pos = (int(self.kf.statePost[0,0]), int(self.kf.statePost[1,0]))

        self.hits = 1      # Consideramos a inicialização como um "hit"
        self.misses = 0
        self.age = 0       # Será incrementado pela primeira chamada a predict()

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
            positions.append((x+30, y-10, 1))
            x += 60
        else:
            x += 10
    return positions

def detect_enemy2(frame):
    lower_orange = np.array([11, 255, 216])
    upper_orange = np.array([7, 255, 255])
    mask = cv2.inRange(frame, lower_orange, upper_orange)
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
                centers.append((cx, cy, 3)) # Tipo 3
    return centers
# --- FIM DAS FUNÇÕES DE DETECÇÃO ---

def closest_enemy(player_pos, enemy_predictions):
    best_idx, best_dist = -1, float('inf')
    px, py = player_pos
    if not enemy_predictions:
        return best_idx, best_dist
    for i, (ex, ey, _) in enumerate(enemy_predictions):
        d = np.hypot(px-ex, py-ey)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx, best_dist

def main():
    global enemy_trackers, next_tracker_id
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    b_press_frames = 0
    press_down = False

    max_misses = 3
    association_threshold = 75

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

        all_current_detections = []
        for x, y, type_val in detections_e1: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False})
        for x, y, type_val in detections_e2: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False}) # Corrigido typo
        for x, y, type_val in detections_e3: all_current_detections.append({'pos': (x,y), 'type': type_val, 'matched': False})

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
            enemies_for_logic.append((tracker.predicted_pos[0], tracker.predicted_pos[1], tracker.type))

        action = np.zeros(env.action_space.shape[0], dtype=np.uint8)

        if enemies_for_logic:
            idx, dist = closest_enemy(player_pos, enemies_for_logic)
            if idx != -1:
                ex, ey, enemy_type = enemies_for_logic[idx]

                if dist < 100: # Distância para atacar
                    if enemy_type == 3:
                        press_down = True
                    else:
                        press_down = False
                    b_press_frames = 3
                    # Movimento enquanto se prepara para atacar
                    action[7 if player_pos[0] < ex else 6] = 1
                else: # Perseguir
                    press_down = False # Garante que não fica abaixado se fora do range de ataque tipo 3
                    action[7 if player_pos[0] < ex else 6] = 1

            else: # Nenhum inimigo próximo encontrado pelos rastreadores (idx == -1)
                 action[7] = 1
        else: # Nenhum inimigo na lista `enemies_for_logic`
            action[7] = 1

        # Lógica de ataque
        if b_press_frames > 0:
            action = np.zeros(env.action_space.shape[0], dtype=np.uint8) # Zera outras ações
            action[0] = 1 # Botão 'B' (ataque)
            b_press_frames -= 1
            if press_down:
                action[5] = 1 # 'DOWN'

        obs, _, done, _ = env.step(action)

        debug_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        px, py = player_pos
        cv2.rectangle(debug_frame, (px-30, py-60), (px+30, py+60), (255, 0, 0), 2) # Desenho do jogador original

        for tracker in enemy_trackers:
            pred_x, pred_y = tracker.predicted_pos
            color = (0,0,0)
            if tracker.type == 1: color = (0, 0, 255)
            elif tracker.type == 2: color = (0, 255, 0)
            elif tracker.type == 3: color = (0, 255, 255)
            cv2.circle(debug_frame, (pred_x, pred_y), 8, color, -1)
            cv2.putText(debug_frame, f"T{tracker.type}", (pred_x + 10, pred_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("Rush'n Attack - Kalman Prediction", debug_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if done:
            obs = env.reset()
            enemy_trackers = []
            next_tracker_id = 0

        time.sleep(dt)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
