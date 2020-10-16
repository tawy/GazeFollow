import sys
import os
import cv2

import argparse

from gf.data import GFImage
from gf.network import GFNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained GazeFollow network on input image")
    parser.add_argument("picture", help="Path to picture")
    parser.add_argument("ex", type=float, help="Position of the face (X-axis, 0.0:left, 1.0:right)")
    parser.add_argument("ey", type=float, help="Position of the face (Y-axis, 0.0:top, 1.0:bottom)")
    parser.add_argument("tx", nargs="?", default=-1.0, type=float, help="Position of the target (X-axis, 0.0:left, 1.0:right)")
    parser.add_argument("ty", nargs="?", default=-1.0, type=float, help="Position of the target (Y-axis, 0.0:top, 1.0:bottom)")
    return parser.parse_args()



def main(args):
    network = GFNetwork()
    picture = GFImage(args.picture)

    if args.ex < 0 or args.ex > 1 or args.ey < 0 or args.ey > 1:
        raise ValueError("The face should be inside the picture (e.g. between 0 and 1)")
    picture.addAnnotation(args.ex, args.ey, args.tx, args.ty) # If tx or ty not in (0,1), it is interpreted as unknown

    network.forward(picture, eye_index=0)

    print(network.gaze)
    for row in network.hm:
        print(("{:.3f} "*15).format(*[float(val) for val in row]))

    picture.addAnnotation(args.ex, args.ey, *network.gaze)
    annotated_picture = picture.getAnnotatedImage()
    cv2.imshow("Result", annotated_picture)
    cv2.waitKey()






if __name__ == "__main__":
    args = parse_args()
    main(args)
