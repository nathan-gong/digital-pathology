import cv2
import numpy as np
import os
from contextlib import contextmanager
from PIL import Image, ImageOps


@contextmanager
def process_image(infile: str, outfile: str = ""):
    try:
        im = Image.open(infile)
        yield im
    finally:
        if not outfile:
            outfile = "{}_modified.jpg".format(os.path.splitext(infile)[0])
        im.save(outfile)


def rotate(infile: str, outfile: str = "", rotations: int = 1):
    """
    rotations: number of times to rotate counterclockwise
    """
    with process_image(infile, outfile) as im:
        while rotations > 0:
            im.transpose(Image.ROTATE_90)
            rotations -= 1


def shift(infile: str, outfile: str = "", x: int = 0, y: int = 0):
    """
    x: pixels to shift in the x-direction
    y: pixels to shift in the y-direction
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    im = cv2.imread(infile)
    (rows, cols) = im.shape[:2]
    im_mod = cv2.warpAffine(im, M, (cols, rows))
    if not outfile:
        outfile = "{}_modified.jpg".format(os.path.splitext(infile)[0])
    cv2.imwrite(outfile, im_mod)


def invert(infile: str, outfile: str = ""):
    with process_image(infile, outfile) as im:
        im = ImageOps.invert(im)


def flip(infile: str, outfile: str = "", flip: bool = True):
    """
    flip: True for horizontal flip, False for vertical flip
    """
    with process_image(infile, outfile) as im:
        if flip:
            im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            im.transpose(Image.FLIP_TOP_BOTTOM)
