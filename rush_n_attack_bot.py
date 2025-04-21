import retro
import numpy as np
import cv2

def find_player(binary_image, block_height=120, block_widght=60):
    height, widght = binary_image.shape
    max_density = 0
    position = (0, 0)
    for y in range(0, height, 10):
        for x in range(0, widght, 10):
            block = binary_image[y:y+block_height, x:x+block_widght]
            density = np.sum(block == 60)
            if density > max_density:
                max_density = density
                position = (x, y)
    return position

def find_enemy_1(binary_image, positions, block_height=60, block_widght=60):
    height, widght = binary_image.shape
    distance = 100
    min_density_white = 250
    min_density_black = 250
    max_density_black = 2000
    max_density_blue = 200
    max_density_gray = 1000
    for y in range(0, height, block_height):
        for x in range(0, widght, block_widght):
            block = binary_image[y:y+block_height, x:x+block_widght]
            density = np.sum(block == 250)
            density_blue = np.sum(block == 60)
            density_gray = np.sum(block == 110)
            if density > min_density_white and density_blue < max_density_blue and density_gray < max_density_gray:
                block = binary_image[y+60:y+60+block_height, x:x+block_widght]
                density = np.sum(block == 0)
                density_blue = np.sum(block == 60)
                density_gray = np.sum(block == 110)
                if density > min_density_black and density < max_density_black and density_blue < max_density_blue and density_gray < max_density_gray:
                    new_pos = (x, y)
                    detected = False
                    for pos in positions:
                        if np.linalg.norm(np.array(new_pos) - np.array(pos)) < distance:
                            detected = True
                    if not detected:
                        positions.append(new_pos)

def closest_enemy(player, positions):
    min = 50000
    min_idx = 0
    for idx, position in enumerate(positions):
        distance = np.linalg.norm(np.array(player) - np.array(position))
        if distance < min:
            min_idx = idx
            min = distance
    return min_idx, min

def main():
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    while True:
        env.render()
        ###### ['B', 'A', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT'] #####
        action = np.zeros(env.action_space.shape[0], dtype=np.uint8)
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (960, 900))
        frame = frame[200:frame.shape[0]-55,:]
        # kernel = np.ones((5, 5), np.uint8)  # Estrutura usada para dilatação (você pode aumentar o tamanho)
        # frame = cv2.dilate(frame, kernel, iterations=1)
        gap = 10
        frame = (frame // gap) * gap
        player_position = find_player(frame)
        enemy_positions = []
        find_enemy_1(frame, enemy_positions)
        x, y = player_position
        ########## Debugging ###########
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(frame, (x, y), (x+60, y+120), (255, 0, 0), 2)
        for (x, y) in enemy_positions:
            cv2.rectangle(frame, (x, y), (x + 60, y + 120), (0, 0, 255), 2)
        ########## Debugging ###########
        if enemy_positions:
            enemy, dist = closest_enemy(player_position, enemy_positions)
            if dist < 80:
                action[0] = 1 # B
            else:
                if player_position[0] < enemy_positions[enemy][0]:
                    #pass
                    action[7] = 1 # Right
                else:
                    #pass
                    action[6] = 1 # Left
        else:
            action[7] = 1 # Right
        #r = 110
        #frame = cv2.inRange(frame, r, r)
        cv2.imshow("Rush'n Attack - NES", frame)
        #action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        #time.sleep(0.02)
        if done:
            obs = env.reset()
        # ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
