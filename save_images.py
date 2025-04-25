import retro
import numpy as np
import cv2
import time
import keyboard  # <-- novo!

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

    print("[INFO] Pressione '1' para capturar o frame 1 e o 2 automaticamente após delay.")
    print("[INFO] Use WASD para movimentar. Tecla 'K' realiza ação.")

    frames_to_wait = 35  # número de frames a esperar entre as capturas

    # Variáveis de controle
    capturing = False
    frames_waited = 0

    while True:
        display_frame = cv2.resize(obs, (640, 480), interpolation=cv2.INTER_LINEAR)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Rush'n Attack - NES", display_frame)
        cv2.waitKey(1)

        # Resetar ações
        action[:] = 0

        # Controles
        if keyboard.is_pressed('a'):
            action[6] = 1
        if keyboard.is_pressed('d'):
            action[7] = 1
        if keyboard.is_pressed('w'):
            action[4] = 1
        if keyboard.is_pressed('s'):
            action[5] = 1
        if keyboard.is_pressed('k'):
            action[0] = 1

        # Captura automática com delay entre os frames
        if keyboard.is_pressed('1') and not capturing and frame1 is None:
            frame1 = preprocess_frame(obs)
            cv2.imwrite("frame_1_manual.png", frame1)
            print("[INFO] Frame 1 capturado.")
            capturing = True
            frames_waited = 0

        if capturing:
            frames_waited += 1
            if frames_waited >= frames_to_wait:
                frame2 = preprocess_frame(obs)
                cv2.imwrite("frame_2_manual.png", frame2)
                print("[INFO] Frame 2 capturado automaticamente após delay.")

                # Processar diferença
                if frame1 is not None and frame2 is not None:
                    diff = cv2.absdiff(frame2, frame1)
                    mask = np.any(diff > 0, axis=2).astype(np.uint8) * 255
                    cv2.imwrite("mask_player_movement.png", mask)

                    mask_rgb = cv2.merge([mask, mask, mask])
                    highlighted = cv2.bitwise_and(frame2, mask_rgb)
                    highlighted = cv2.erode(highlighted, np.ones((3, 3), np.uint8), iterations=1)

                    # Criar canal alfa com base na máscara (255 onde há movimento, 0 onde não há)
                    alpha_channel = mask.copy()

                    # Juntar canais RGB + Alpha
                    highlighted_rgba = cv2.merge((highlighted[:, :, 0], highlighted[:, :, 1], highlighted[:, :, 2], alpha_channel))

                    # Salvar com transparência
                    cv2.imwrite("player_crop.png", highlighted_rgba)

                    save_histograms_rgb(diff, "player_rgb")
                    print("[INFO] Crop salvo em player_crop.png e histogramas RGB salvos.")

                # Reset para nova captura
                capturing = False
                frame1 = None
                frame2 = None

        # Sair
        if keyboard.is_pressed('q'):
            print("[INFO] Encerrando...")
            break

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

        time.sleep(1 / 60)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
