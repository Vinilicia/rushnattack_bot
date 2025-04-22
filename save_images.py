import retro
import numpy as np
import cv2

def save_histograms_rgb(image, prefix):
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
    np.save(f"{prefix}_b.npy", hist_b)
    np.save(f"{prefix}_g.npy", hist_g)
    np.save(f"{prefix}_r.npy", hist_r)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (960, 900))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[200:frame.shape[0]-55, :]
    return frame

def main():
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    action = np.zeros(env.action_space.shape[0], dtype=np.uint8)

    frame1 = None
    frame2 = None

    print("[INFO] Jogue e pressione '1' para capturar o primeiro frame, '2' para o segundo, 'q' para sair.")
    print("[INFO] Use as teclas WASD para controlar o personagem: W = Pular, A = Esquerda, D = Direita, S = Agachar.")

    while True:

        key = cv2.waitKey(1) & 0xFF

        # Controles para o jogador
        if key == ord('a'):  # Move para a esquerda (action[6] -> LEFT)
            action[6] = 1  # Ação para mover para a esquerda
            action[7] = 0  # Sem movimento para a direita
        elif key == ord('d'):  # Move para a direita (action[7] -> RIGHT)
            action[7] = 1  # Ação para mover para a direita
            action[6] = 0  # Sem movimento para a esquerda
        elif key == ord('w'):  # Pular (action[4] -> UP)
            action[4] = 1  # Ação para pular
        elif key == ord('s'):  # Agachar (action[5] -> DOWN)
            action[5] = 1  # Ação para agachar
        elif key == ord('k'):  # Agachar (action[5] -> DOWN)
            action[0] = 1  # Ação para agachar
        elif key == ord('1'):  # Captura o primeiro frame
            frame1 = preprocess_frame(obs)
            cv2.imwrite("frame_1_manual.png", frame1)
            print("[INFO] Frame 1 capturado.")
        elif key == ord('2'):  # Captura o segundo frame
            frame2 = preprocess_frame(obs)
            cv2.imwrite("frame_2_manual.png", frame2)
            print("[INFO] Frame 2 capturado.")

            if frame1 is not None:
                diff = cv2.absdiff(frame2, frame1)
                mask = np.any(diff > 0, axis=2).astype(np.uint8) * 255
                cv2.imwrite("mask_player_movement.png", mask)

                mask_rgb = cv2.merge([mask, mask, mask])
                highlighted = cv2.bitwise_and(frame2, mask_rgb)
                highlighted = highlighted[:, 300:600, :]
                highlighted = cv2.erode(highlighted, np.ones((3, 3), np.uint8), iterations=1)

                cv2.imwrite("player_crop.png", highlighted)
                save_histograms_rgb(diff, "player_rgb")
                print("[INFO] Crop salvo em player_crop.png e histogramas RGB salvos.")
        elif key == ord('q'):  # Sair do jogo
            break
        else:
            # Resetar as ações caso nenhuma tecla seja pressionada
            action[6] = 0  # Sem movimento para a esquerda
            action[7] = 0  # Sem movimento para a direita
            action[4] = 0  # Sem pular
            action[5] = 0  # Sem agachar
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
        env.render()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
