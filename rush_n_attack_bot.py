import retro
import numpy as np
import time
import cv2

def encontrar_player(imagem_binaria, bloco_altura=120, bloco_largura=60):
    altura, largura = imagem_binaria.shape
    max_densidade = 0
    posicao_max = (0, 0)
    for y in range(0, altura, 10):
        for x in range(0, largura, 10):
            bloco = imagem_binaria[y:y+bloco_altura, x:x+bloco_largura]
            densidade = np.sum(bloco == 60)
            if densidade > max_densidade:
                max_densidade = densidade
                posicao_max = (x, y)
    return posicao_max

def encontrar_inimigo_1(imagem_binaria, bloco_altura=60, bloco_largura=60):
    altura, largura = imagem_binaria.shape
    distancia = 80
    min_densidade_branco = 200
    min_densidade_preto = 60
    posicoes = []
    for y in range(0, altura, bloco_altura):
        for x in range(0, largura, bloco_largura):
            bloco = imagem_binaria[y:y+bloco_altura, x:x+bloco_largura]
            densidade = np.sum(bloco == 250)
            if densidade > min_densidade_branco:
                bloco = imagem_binaria[y+60:y+60+bloco_altura, x:x+bloco_largura]
                densidade = np.sum(bloco == 0)
                if densidade > min_densidade_preto:
                    pos_nova = (x, y)
                    ja_detectado = False
                    for pos in posicoes:
                        if np.linalg.norm(np.array(pos_nova) - np.array(pos)) < distancia:
                            ja_detectado = True
                    if not ja_detectado:
                        posicoes.append(pos_nova)
    return posicoes

def main():
    env = retro.make(game="RushnAttack-Nes")
    obs = env.reset()
    while True:
            env.render()
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (960, 900))
            frame = frame[200:frame.shape[0]-55,:]
            intervalo = 10
            frame = (frame // intervalo) * intervalo
            pos = encontrar_player(frame)
            posicoes = encontrar_inimigo_1(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            x, y = pos
            cv2.rectangle(frame, (x, y), (x+60, y+120), (255, 0, 0), 2)
            for (x, y) in posicoes:
                cv2.rectangle(frame, (x, y), (x + 60, y + 120), (0, 0, 255), 2)
            cv2.imshow("Rush'n Attack - NES", frame)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #time.sleep(0.02)
            if done:
                obs = env.reset()
            # Pressione ESC para sair
            if cv2.waitKey(1) & 0xFF == 27:
                break
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
