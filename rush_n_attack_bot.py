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

def find_enemy_1(binary_image, block_height=60, block_widght=60):
    height, widght = binary_image.shape
    distance = 80
    min_density_white = 200
    min_density_black = 60
    positions = []
    for y in range(0, height, block_height):
        for x in range(0, widght, block_widght):
            block = binary_image[y:y+block_height, x:x+block_widght]
            density = np.sum(block == 250)
            if density > min_density_white:
                block = binary_image[y+60:y+60+block_height, x:x+block_widght]
                density = np.sum(block == 0)
                if density > min_density_black:
                    new_pos = (x, y)
                    detected = False
                    for pos in positions:
                        if np.linalg.norm(np.array(new_pos) - np.array(pos)) < distance:
                            detected = True
                    if not detected:
                        positions.append(new_pos)
    return positions

def main():
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    while True:
            env.render()
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (960, 900))
            frame = frame[200:frame.shape[0]-55,:]
            gap = 10
            frame = (frame // gap) * gap
            pos = find_player(frame)
            positions = find_enemy_1(frame)
            x, y = pos
            ########## Debugging ###########
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(frame, (x, y), (x+60, y+120), (255, 0, 0), 2)
            for (x, y) in positions:
                cv2.rectangle(frame, (x, y), (x + 60, y + 120), (0, 0, 255), 2)
            ########## Debugging ###########
            cv2.imshow("Rush'n Attack - NES", frame)
            action = env.action_space.sample()
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
