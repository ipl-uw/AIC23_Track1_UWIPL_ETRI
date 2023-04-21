import argparse
import csv
import os
import sys
import numpy as np              # conda install numpy
import pickle
import csv

from loguru import logger       # conda install loguru
sys.path.append('.')

def make_parser():
    parser = argparse.ArgumentParser("bbox removal")
    parser.add_argument("root_path", default="/home/hsiangwei/Desktop/AICITY2023/data", type=str)
    return parser

def validate1(left, right, top, bottom, margin):
    return left >= margin and right <= 1920 - margin and top >= margin

def isTopTouch(top, margin):
    return top <= margin
    
def isSideTouch(left, right, margin):
    return left <= margin or right >= 1920 - margin
    
def isBottomTouch(bottom, margin):
    return bottom >= 1080 - margin

def isValidAR(left, right, top, bottom, minar, maxar):
    left = max(0, left)
    right = min(1920, right)
    top = max(0, top)
    bottom = min(1080, bottom)

    width = right - left
    height = bottom - top
    ar = width / height

    return ar >= minar and ar <= maxar


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def main():
    args = make_parser().parse_args()
    root_path = args.root_path
    margin = 2
    minar = 0.3
    maxar = 1.6
    valid_save = True
    file_path = os.path.join(root_path, 'final_n=15_dist200')
    file_output_path = os.path.join(root_path, 'final_n=15_dist200_pk_filter_margin_2')
    os.makedirs(file_output_path,exist_ok=True)
    labels = os.listdir(file_path)
    labels.sort()
    
    for label in labels:
        f = open(os.path.join(file_path, label))
        rdr = csv.reader(f)
        incount = 0
        validlines = []
        removedlines = []
        for line in rdr:
            left = round(float(line[2]))
            top = round(float(line[3]))
            right = left + round(float(line[4]))
            bottom = top + round(float(line[5]))
            processed = False
            if isTopTouch(top, margin):
                processed = True
                removedlines.append(line)
            else:
                if isSideTouch(left, right, margin) or isBottomTouch(bottom, margin):
                    if not isValidAR(left, right, top, bottom, minar, maxar):
                        removedlines.append(line)
                        processed = True

            if not processed:
                validlines.append(line)

            incount += 1

        print(label, incount, len(validlines), sep=',')

        if valid_save:
            createDirectory(file_output_path)
            fo = open(os.path.join(file_output_path, label), 'w', newline='')
            wtr = csv.writer(fo)
            wtr.writerows(validlines)

if __name__ == "__main__":
    main()