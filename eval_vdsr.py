import argparse, os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from VDSR import Net
import numpy as np
import math
import cv2
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Eval VDSR")
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--imgpath', default='/data/Test/Set5/baby_GT_scale_3.bmp', type=str)
    opt = parser.parse_args() # opt < parser
    print(opt)

    model = Net() # net
    print("===> load model")
    criterion = nn.MSELoss(size_average=False) # set loss

    checkpoint = torch.load(opt.pretrained)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loading model '{}'".format(opt.pretrained))

    model.eval()

    img_path = os.path.join(os.getcwd(), opt.imgpath)
    image = cv2.imread(img_path)
    print("=> loading image '{}'".format(opt.imgpath))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image_Y = image[:, :, 0]



if __name__ == "__main__":
    main()
