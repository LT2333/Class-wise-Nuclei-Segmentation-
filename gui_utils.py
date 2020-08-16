import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from model import UNet2
import matplotlib.pylab as plt
import shutil

from dataset import get_test_loader
from train import run_test
from utils import Option, save_imgs


def model_start(image_path):
    opt = Option()
    model = UNet2(input_channels=3, nclasses=1)
    model_cls = UNet2(input_channels=3, nclasses=8)
    # model.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'model-01.pt'), map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("../" + os.path.join(opt.checkpoint_dir, 'model-120-gray.pt')))
        model_cls.load_state_dict(torch.load("../" + os.path.join(opt.checkpoint_dir, 'model-150-color.pt')))
    else:
        model.load_state_dict(torch.load("../" + os.path.join(opt.checkpoint_dir, 'model-120-gray.pt'), map_location=torch.device('cpu')))
        model_cls.load_state_dict(torch.load("../" + os.path.join(opt.checkpoint_dir, 'model-150-color.pt'), map_location=torch.device('cpu')))

    if not os.path.exists("temp"):
        os.mkdir("temp")
    if not os.path.exists("temp/folder01"):
        os.mkdir("temp/folder01")
    # shutil.copy(image_path, 'temp/folder01/')
    os.rename(image_path, 'temp/folder01/image.png')

    test_loader = get_test_loader("temp", batch_size=opt.batch_size, shuffle=opt.shuffle,
                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    predictions, img_ids = run_test(model, test_loader, opt,  mode='gray')
    plt.imsave(os.path.join("temp", 'prediction.png'), predictions[0], cmap='gray')

    predictions_cls, _ = run_test(model_cls, test_loader, opt,  mode='colored')
    print(predictions_cls[0].shape)
    print(os.path.join('prediction_cls.png'))
    plt.imsave(os.path.join("temp", 'prediction_cls.png'), predictions_cls[0], vmin=0, vmax=7, cmap='jet')

    # plt.imsave(os.path.join("temp", 'prediction_cls.png'), predictions_cls[0])

    ############
    overlay = convert_overlay()
    ############

    # return 'temp/prediction.png.'
    return overlay


def convert_overlay():
    # Load image, create mask, and draw white circle on mask
    image = cv2.imread("temp/folder01/image.png")
    mask = cv2.imread('temp/prediction.png', 0)
    greenImg = np.zeros(image.shape, image.dtype)

    greenImg[:, :] = (0, 224, 0)
    redMask = cv2.bitwise_and(greenImg, greenImg, mask=mask)
    result = cv2.addWeighted(redMask, 1, image, 1, 0, mask)

    # mask = cv2.imread('temp/prediction.png', -1)
    print(image.shape)
    print(mask.shape)
    # result = cv2.addWeighted(mask, 10, image, 0.8, 10)
    plt.imsave(os.path.join("temp", "overlay.png"), result)
    return ['temp/overlay.png', 'temp/prediction_cls.png']

def remove_temp_dir(image_path):
    os.rename('temp/folder01/image.png', image_path)
    shutil.rmtree("temp")


# if __name__ == '__main__':
#     model_start(image_path)