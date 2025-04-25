import cv2
import numpy as np

def save_combined_hsv_histogram(image_path, output_path):
    # Carrega a imagem com canal alfa
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Imagem '{image_path}' não encontrada.")

    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        bgr_image = cv2.merge([b, g, r])
        mask = a > 0
    else:
        bgr_image = image
        mask = None

    # Converte para HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Cria máscara binária se necessário
    mask_uint8 = mask.astype(np.uint8) if mask is not None else None

    # Calcula histogramas HSV
    hist_h = cv2.calcHist([hsv_image], [0], mask_uint8, [180], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_image], [1], mask_uint8, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_image], [2], mask_uint8, [256], [0, 256]).flatten()

    hist_h = hist_h / hist_h.sum() if hist_h.sum() > 0 else hist_h
    hist_s = hist_s / hist_s.sum() if hist_s.sum() > 0 else hist_s
    hist_v = hist_v / hist_v.sum() if hist_v.sum() > 0 else hist_v

    # Calcula a média HSV apenas dos pixels válidos
    if mask is not None:
        valid_pixels = hsv_image[mask]
    else:
        valid_pixels = hsv_image.reshape(-1, 3)

    mean_hsv = np.mean(valid_pixels, axis=0) if len(valid_pixels) > 0 else [0, 0, 0]

    # Salva tudo
    np.savez(output_path, H=hist_h, S=hist_s, V=hist_v, mean_H=mean_hsv[0], mean_S=mean_hsv[1], mean_V=mean_hsv[2])
    print(f"[INFO] Histograma HSV + médias HSV salvos em: {output_path}.npz")

# Exemplo com suas imagens
save_combined_hsv_histogram("images/enemy.png", "enemy_hist_hsv")
save_combined_hsv_histogram("images/enemy2.png", "enemy2_hist_hsv")
save_combined_hsv_histogram("images/enemy3.png", "enemy3_hist_hsv")
