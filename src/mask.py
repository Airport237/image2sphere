import cv2
import numpy as np
from pathlib import Path


def segment_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def apply_mask(img, mask):
    result = cv2.bitwise_and(img, img, mask=mask)
    bg = np.ones_like(img) * 0
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return (bg * (1 - mask_3ch) + result * mask_3ch).astype(np.uint8)


def process_image(input_path, output_path):
    img = cv2.imread(str(input_path))
    mask = segment_contour(img)
    result = apply_mask(img, mask)
    cv2.imwrite(str(output_path), result)
    return result, mask


def main():
    assets = Path("assets")
    output = Path("output")
    output.mkdir(exist_ok=True)

    images = list(assets.glob("crop*")) + list(assets.glob("t2.*"))

    for img_path in sorted(images):
        out_path = output / f"{img_path.stem}_contour.png"
        result, mask = process_image(img_path, out_path)
        print(f"{img_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
