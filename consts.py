import numpy as np

ym_per_pix, xm_per_pix = 30 / 720, 3.7 / 700  # meters per pixel in x dimension

yellow_hsv_min = np.array([20, 70, 70])
yellow_hsv_max = np.array([50, 255, 255])