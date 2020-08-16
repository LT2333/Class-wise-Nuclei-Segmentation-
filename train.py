import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import get_train_valid_loader, get_test_loader
from model import UNet2, UNet
from utils import Option, compute_iou, save_imgs, compute_accuracy, compute_other_scores
from skimage.transform import resize
import torch

def train(model, train_loader, opt, criterion, epoch):
    model.train()
    num_batches = 0
    avg_loss = 0
    with open('train_logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(train_loader):
            data = sample_batched['image']
            target = sample_batched['mask']

            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype)).long()
            optimizer.zero_grad()
            output = model(data)
            target = torch.squeeze(target,1)
            # print(target.shape)
            # print(output.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            num_batches += 1
        avg_loss /= num_batches
        # avg_loss /= len(train_loader.dataset)
        print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss) + '\n')

def val(model, val_loader, opt, criterion, epoch):
    model.eval()
    num_batches = 0
    avg_loss = 0
    with open('val_logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(val_loader):
            data = sample_batched['image']
            target = sample_batched['mask'].long()
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype)).long()
            output = model.forward(data)
            target = torch.squeeze(target, 1)
            # print(target)
            # print(output)
            loss = criterion(output, target)
            avg_loss += loss.item()
            num_batches += 1
        avg_loss /= num_batches
        # avg_loss /= len(val_loader.dataset)

        print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss) + '\n')

# train and validation
def run(model, train_loader, val_loader, opt, criterion):
    for epoch in range(1, opt.epochs):
        train(model, train_loader, opt, criterion, epoch)
        val(model, val_loader, opt, criterion, epoch)
        predictions, img_ids = run_test(model, val_loader, opt)
        compute_accuracy(predictions, img_ids)
        compute_iou(predictions, img_ids)
    compute_other_scores(predictions, img_ids)

# only train
def run_train(model, train_loader, opt, criterion):
    for epoch in range(1, opt.epochs):
        train(model, train_loader, opt, criterion, epoch)

# make prediction
def run_test(model, test_loader, opt, mode='colored'):
    model.eval()
    predictions = []
    img_ids = []
    for batch_idx, sample_batched in enumerate(test_loader):
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], sample_batched['width']
        data = Variable(data.type(opt.dtype))
        output = model.forward(data)

        output = output.data.cpu().numpy()

        output = output.transpose((0, 2, 3, 1))    # transpose to (B,H,W,C)

        for i in range(0,output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            h = height[i]
            w = width[i]
            if not isinstance(h, int):
                h = h.cpu().numpy()
                w = w.cpu().numpy()
            pred_mask = resize(pred_mask, (h, w), mode='constant')
            if mode == 'colored':
                mask_copy = np.argmax(pred_mask,axis=2)
                # print(mask_copy.shape)
                # print(mask_copy)
                predictions.append(mask_copy)
                img_ids.append(id)
            else:
                pred_mask = (pred_mask > 0.5)
                predictions.append(pred_mask)
                img_ids.append(id)


    return predictions, img_ids

if __name__ == '__main__':
    """Train Unet model"""
    opt = Option()
    # model = UNet(input_channels=3, nclasses=8)
    if opt.is_colored:
        model = UNet2(input_channels=3, nclasses=8)
    else:
        model = UNet2(input_channels=3, nclasses=1)

    if opt.is_train:
        # split all data to train and validation, set split = True
        train_loader, val_loader = get_train_valid_loader(opt.root_dir, batch_size=opt.batch_size,
                                              split=True, shuffle=opt.shuffle,
                                              num_workers=opt.num_workers,
                                              val_ratio=0.1, pin_memory=opt.pin_memory)

        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        if opt.is_cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        if opt.is_colored:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.BCELoss().cuda()
        # start to run a training only
        # run_train(model, train_loader, opt, criterion)
        # run training together with val
        run(model, train_loader,val_loader, opt, criterion)

        # SAVE model
        if opt.save_model:
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, 'model-{}-unet1.pt'.format(opt.epochs)))
    else:
        # load testing data for making predictions
        test_loader = get_test_loader(opt.test_dir, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                      num_workers=opt.num_workers, pin_memory=opt.pin_memory)
        # load the model and run test
        if opt.is_colored:
            model.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'model-150-color.pt')))
        else:
            model.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'model-120-gray.pt')))
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        if opt.is_cuda:
            model = model.cuda()
        if opt.is_colored:
            predictions, img_ids = run_test(model, test_loader, opt, mode='colored')
        else:
            predictions, img_ids = run_test(model, test_loader, opt, mode='gray')
        compute_iou(predictions, img_ids)
        compute_accuracy(predictions, img_ids)
        compute_other_scores(predictions, img_ids)
        save_imgs(predictions,img_ids)


