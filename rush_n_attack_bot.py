import retro
import cv2
import numpy as np
import time

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
    #for y in(365, 570):
    y = 570
    x = 0
    while x < width:
        if x + block_width > width:
            break
        block = frame[y:y+block_height, x:x+block_width]
        mask_white = np.all(block >= white, axis=2)
        sum_white = np.sum(mask_white)
        if sum_white > min_white:
            positions.append((x+30, y-10, 1))
            x += 60
        else:
            x += 10
    return positions

def detect_enemy2(frame):
    lower_orange = np.array([5, 250, 200])
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
                centers.append((cx, cy, 1))
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
                centers.append((cx, cy, 3))
    return centers

def closest_enemy(player_pos, enemy_positions):
    best_idx, best_dist = 0, float('inf')
    px, py = player_pos
    for i, (ex, ey, _) in enumerate(enemy_positions):
        d = np.hypot(px-ex, py-ey)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx, best_dist

def main():
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    b_press_frames = 0
    press_down = False

    while True:
        ###### ['B', 'A', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT'] #####
        obs = cv2.resize(obs, (960, 900))[200:900-50, :]
        frame = obs
        enemies1 = detect_enemy1(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        player_pos = detect_player(frame)
        enemies2 = detect_enemy2(frame)
        enemies3 = detect_enemy3(frame)
        enemies = enemies1 + enemies2 + enemies3
        action = np.zeros(env.action_space.shape[0], dtype=np.uint8)
        if enemies:
            idx, dist = closest_enemy(player_pos, enemies)
            ex, ey, type = enemies[idx]
            if dist < 100:
                if type == 3:
                    press_down = True
                else:
                    press_down = False
                b_press_frames = 3
            else:
                action[7 if player_pos[0] < ex else 6] = 1
        else:
            action[7] = 1  # direita

        if b_press_frames > 0:
            action = np.zeros(env.action_space.shape[0], dtype=np.uint8)
            action[0] = 1
            b_press_frames -= 1
            if press_down:
                action[5] = 1
        obs, _, done, _ = env.step(action)

        debug = frame.copy()
        debug = cv2.cvtColor(debug, cv2.COLOR_HSV2BGR)
        px, py = player_pos
        cv2.rectangle(debug, (px-30, py-60), (px+30, py+60), (255, 0, 0), 2)
        for (ex, ey, _) in enemies1:
           cv2.rectangle(debug, (ex-10, 510), (ex+50, 620), (0, 0, 255), 2)
        for (ex, ey, _) in enemies2:
           cv2.rectangle(debug, (ex-30, ey-60), (ex+30, ey+60), (0, 255, 0), 2)
        for (ex, ey, _) in enemies3:
           cv2.rectangle(debug, (ex-30, ey-60), (ex+30, ey+60), (0, 255, 255), 2)

        cv2.imshow("Rush'n Attack - HSV Detection", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if done:
            obs = env.reset()
        time.sleep(1/60)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
