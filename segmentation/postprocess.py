import cv2
import numpy as np


def cv_fill(mask):
    mask = (mask > 0).astype(np.uint8)
    img = (mask * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel, iterations=1)

    im_floodfill = img1.copy()
    h, w = img1.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, flood_mask, (0,0), 255)

    inv = cv2.bitwise_not(im_floodfill)
    filled = img1 | inv

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)

    clean = np.zeros_like(filled)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 200:
            clean[labels == i] = 255

    return (clean > 0).astype(np.uint8)


def smooth_thicken(mask):
    img = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    smooth = np.zeros_like(img)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 100:
            continue

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        cv2.drawContours(smooth, [approx], -1, 255, thickness=6)
        cv2.drawContours(smooth, [approx], -1, 255, thickness=-1)

    smooth = cv2.GaussianBlur(smooth, (3,3), 0)
    _, smooth = cv2.threshold(smooth, 80, 255, cv2.THRESH_BINARY)

    return (smooth > 0).astype(np.uint8)
