import sys
import os
import cv2

import argparse

from gf.data import GFReader
from gf.network import GFNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained GazeFollow network on input image")
    parser.add_argument("path", help="Path to gazefollow dataset (file train_annotations.txt)")
    return parser.parse_args()


def main(args):
    network = GFNetwork()
    reader = GFReader(args.path)

    key = 0
    while key != 27:
        picture = reader.getRandom()
        network.forward(picture, eye_index=0)
        picture.addAnnotation(*picture.annotations[0].eyes, *network.gaze, color=(0,0,255))
        img = picture.getAnnotatedImage()
        cv2.imshow("Result", img)
        key = cv2.waitKey() & 0xFF




if __name__ == "__main__":
    args = parse_args()
    main(args)
