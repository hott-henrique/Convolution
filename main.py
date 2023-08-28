import argparse

import cv2 as ocv
import numpy as np


Sobel = {
    "x": np.array([
        [ +1, 0, -1 ],
        [ +2, 0, -2 ],
        [ +1, 0, -1 ]
    ], dtype=np.float64),

    "y": np.array([
        [ +1, +2, +1 ],
        [  0,  0,  0 ],
        [ -1, -2, -1 ]
    ], dtype=np.float64),
}

def create_extended_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape

    extended_img = np.zeros(shape=(h+2, w+2), dtype=np.float64)

    extended_img[1:h+1, 1:w+1] += img

    extended_img[0, 1:w+1] += img[0]
    extended_img[h+1, 1:w+1] += img[h-1]

    extended_img[1:h+1, 0] += img[:, 0]
    extended_img[1:h+1, w+1] += img[:, w-1]

    return extended_img

def convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape

    extended_img = create_extended_image(img)

    out_img = np.zeros(shape=(h, w), dtype=np.float64)

    for i in range(1, h):
        for j in range(1, w):
            out_img[i-1, j-1] = np.sum(extended_img[i-1:i+2, j-1:j+2] * kernel)

    return out_img

def main(in_img_path: str, out_img_path: str):
    img = ocv.cvtColor(ocv.imread(in_img_path), ocv.COLOR_BGR2GRAY)

    sx_img = convolution(img, Sobel["x"])

    sy_img = convolution(img, Sobel["y"])

    final_img = np.sqrt(np.float_power(sx_img, 2) + np.float_power(sy_img, 2))

    presentation = np.concatenate((sx_img, sy_img, final_img), axis=1)

    ocv.imwrite(out_img_path, presentation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-input", help="Path to read image.", required=True)
    parser.add_argument("--image-output", help="Path to save image.", required=True)

    args = parser.parse_args()

    main(in_img_path=args.image_input, out_img_path=args.image_output)

