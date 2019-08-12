import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net
from dataset_h5 import Read_dataset_h5
import numpy as np
import math

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=80, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--optimizer", default="SGD", type=str, help="set optimizer (default: SGD)")

def main():
    global opt, model #opt, model global
    opt = parser.parse_args() # opt < parser
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus # set gpu
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed) # set seed
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True # find optimal algorithms for hardware

    print("===> Loading datasets")
    train_set = Read_dataset_h5("data/train.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True) # read to DataLoader

    print("===> Building model")
    model = Net() # net
    criterion = nn.MSELoss(size_average=False) # set loss

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            model = torch.load(opt.pretrained)['model']# load model
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda() # set model&loss for use gpu

    # optionally set start_epoch
    if opt.start_epoch:
        if os.path.isfile(opt.start_epoch): # if user input start_epoch
            print("=> start epoch '{}'".format(opt.start_epoch))

    print("===> Setting Optimizer")
    if opt.optimizer == 'SGD':
         optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'Adam':
        if opt.lr == 0.1:
            optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        print("=> input 'SGD' or 'Adam', not '{}'".format(opt.optimizer))

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def PSNR(pred, gt, shave_border=0):
    pred = pred.detach().cpu().numpy().astype(float)
    gt = gt.detach().cpu().numpy().astype(float)
    height, width = pred.shape[-2:]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1 / rmse)

def train(training_data_loader, optimizer, model, criterion, epoch):
    if opt.optimizer == 'SGD':
        lr = adjust_learning_rate(optimizer, epoch-1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            label = label.cuda()
        output = model(input)
        psnr = PSNR(output,label)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),opt.clip)
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): PSNR : {:.10f}".format(epoch, iteration, len(training_data_loader), psnr))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "VDSR_{}_epoch_{}.pth".format(opt.optimizer, epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
