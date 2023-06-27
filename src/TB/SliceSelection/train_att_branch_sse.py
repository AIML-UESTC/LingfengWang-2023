from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from CaseSliceData import SliceData
import ssenet_resnet

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch+1)
                    resumed = True
                else:
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    idx = torch.LongTensor(labels).view(-1, 1)
                    label_one_hot = torch.zeros(len(labels), 2).scatter_(1, idx, 1)
                    label_one_hot = Variable(label_one_hot.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                att_outputs, per_outputs, _ = model(inputs)
                _, preds = torch.max(per_outputs.data, 1)

                #loss = criterion(per_outputs, label_one_hot)
                att_loss = criterion(att_outputs, label_one_hot)
                per_loss = criterion(per_outputs, label_one_hot)
                loss = att_loss + per_loss
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i+1)*args.batch_size)
                batch_acc = running_corrects.cpu().numpy() / ((i+1)*args.batch_size)

                if phase == 'train' and i%args.print_freq == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_lr()[0], phase, batch_loss, batch_acc, \
                        args.print_freq/(time.time()-tic_batch)))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.cpu().numpy() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        if (epoch+1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_papath, "epoch_" + str(epoch) + ".pth.tar"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="checkpoint")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="SSe_resnet_18", help="")
    args = parser.parse_args()

    # read data
    dataloders, dataset_sizes = SliceData(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])

    model = getattr(ssenet_resnet, args.network)(num_classes=args.num_class)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        torch.backends.cudnn.enabled = False
        #model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.BCELoss() #Use sigmoid and BCE loss function
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=50, eta_min=0)
    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=scheduler,
                           num_epochs=args.num_epochs,
                           dataset_sizes=dataset_sizes)
