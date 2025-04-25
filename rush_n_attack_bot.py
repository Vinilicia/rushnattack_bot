import retro
import numpy as np
import cv2

def find_player(bgr_image, block_height=120, block_width=60):
    h, w, _ = bgr_image.shape
    max_density = 0
    best_pos = (0, 0)

    # Ajuste conforme a cor do player
    lower_blue = np.array([100, 50, 50])   # BGR
    upper_blue = np.array([180, 120, 90])

    mask = cv2.inRange(bgr_image, lower_blue, upper_blue)

    for y in range(0, h - block_height + 1, 10):
        for x in range(0, w - block_width + 1, 10):
            block = mask[y:y+block_height, x:x+block_width]
            density = int(block.sum() / 255)
            if density > max_density:
                max_density = density
                best_pos = (x, y)

    return best_pos

def detect_enemy_by_mean_hsv(bgr_image, mean_hsv_target, threshold=30):
    detected = []
    h, w, _ = bgr_image.shape
    bh, bw = 60, 120

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    for y in range(0, h - bh + 1, 60):
        for x in range(0, w - bw + 1, 30):
            block = hsv_image[y:y+bh, x:x+bw]
            mean_block = np.mean(block.reshape(-1, 3), axis=0)
            dist = np.linalg.norm(mean_block - mean_hsv_target)
            if dist < threshold:
                detected.append((x, y))

    return detected

def closest_enemy(player_pos, enemy_positions):
    best_idx, best_dist = 0, float('inf')
    px, py = player_pos
    for i, (ex, ey) in enumerate(enemy_positions):
        d = np.hypot(px-ex, py-ey)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx, best_dist

def main():
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()

    # Carrega o arquivo com médias HSV
    enemy_data = np.load("enemy_hist_hsv.npz")
    mean_hsv = np.array([
        enemy_data["mean_H"],
        enemy_data["mean_S"],
        enemy_data["mean_V"]
    ], dtype=np.float32)

    while True:
        rgb = obs
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (960, 900))[200:900-55, :]

        player_pos = find_player(bgr)
        enemies = detect_enemy_by_mean_hsv(bgr, mean_hsv, threshold=30)

        action = np.zeros(env.action_space.shape[0], dtype=np.uint8)
        if enemies:
            idx, dist = closest_enemy(player_pos, enemies)
            ex, ey = enemies[idx]
            if dist < 80:
                action[0] = 1  # B
            else:
                action[7 if player_pos[0] < ex else 6] = 1
        else:
            action[7] = 1  # direita

        debug = bgr.copy()
        px, py = player_pos
        cv2.rectangle(debug, (px, py), (px+60, py+120), (255, 0, 0), 2)
        for (ex, ey) in enemies:
            cv2.rectangle(debug, (ex, ey), (ex+60, ey+120), (0, 0, 255), 2)

        cv2.imshow("Rush'n Attack - HSV Detection", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
