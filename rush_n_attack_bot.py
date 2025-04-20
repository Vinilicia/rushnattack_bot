import retro
import random
import time

def main():
    # Nome do jogo conforme registrado pelo retro (veja a saída do import: RushnAttack-Nes)
    env = retro.make(game="RushnAttack-Nes")

    obs = env.reset()
    total_reward = 0

    for step in range(1000):  # número de steps (pode ajustar)
        env.render()

        # Escolhe uma ação aleatória válida
        action = env.action_space.sample()

        # Aplica a ação
        obs, reward, done, info = env.step(action)
        total_reward += reward

        time.sleep(0.01)  # de leve pra suavizar

        if done:
            print("Episode finished!")
            break

    env.close()
    print("Total reward:", total_reward)

if __name__ == "__main__":
    main()
