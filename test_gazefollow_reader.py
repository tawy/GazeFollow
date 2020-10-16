import sys
import os
import cv2

from gf.data import GFReader


def main():
    reader = GFReader(sys.argv[1])

    # print(len(reader.images))
    # print(reader.path)
    # print(reader.basepath)

    key = 0
    while key != 27:
        img_cont = reader.getRandom()
        img = img_cont.getAnnotatedImage()
        cv2.imshow(img_cont.path, img)
        key = cv2.waitKey() & 0xFF
        cv2.destroyWindow(img_cont.path)




if __name__ == "__main__":
    main()
