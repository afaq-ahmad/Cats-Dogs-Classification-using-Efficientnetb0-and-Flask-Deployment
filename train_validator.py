import argparse
import torch
import torch.nn as nn
import time
import random
import torch.backends.cudnn as cudnn
import warnings
import numpy as np
import shutil
import glob
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser(description='Model Training or validation on Testing Dataset')

parser.add_argument('--train_dir', default='dataset/training_set/', type=str,help='Training dataset Path')
parser.add_argument('--test_dir', default='dataset/test_set/', type=str,help='Testing dataset Path')
                    
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--imz', '--image_size', default=224, type=int,
                    metavar='IMZ', help='image size for training and testing', dest='image_size')
                    
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--only_validation',default='', type=str,
                    help='For doing only validationpath to latest checkpoint (default: none)')
                    
parser.add_argument('--seed_n', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


def train(train_loader, model, criterion, optimizer, epoch, gpu=0):
    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1,_,_ = accuracy(output, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Epoch: [{}/{}]".format(i,epoch),"   Loss: {}".format(np.round(loss.item(),3)),"   Accuracy: {}".format(np.round(acc1[0].tolist()[0],3)),"   Time(sec): {}".format(np.round(time.time() - end,3)))
        # measure elapsed time
        end = time.time()

def validate(val_loader, model, criterion, gpu=0):
    Prediction_all=[]
    Groundt_all=[]
    # switch to evaluate mode
    model.eval()
    Loss_s=[]
    Acc_s=[]
    Time_s=[]
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1,targetb,predb = accuracy(output, target)
            Loss_s.append(loss.item())
            Acc_s.append(acc1[0].tolist())
            Time_s.append(time.time() - end)
            
            Groundt_all=Groundt_all+targetb
            Prediction_all=Prediction_all+predb
            
            if i % 10 == 0:
                
                print("Validation Batch: [{}]".format(i),"   Loss: {}".format(np.round(loss.item(),3)),"   Accuracy: {}".format(np.round(acc1[0].tolist()[0],3)),"   Time(sec): {}".format(np.round(time.time() - end,3)))
            # measure elapsed time
            end = time.time()
    return np.average(Acc_s),Groundt_all,Prediction_all

def save_checkpoint(state, is_best, filename='weight/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'weight/model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res,target.tolist(),pred[0].tolist()

def calculate_precision_recall_fscore(Groundt_all,Prediction_all):
    precision,recall,fbeta_score,_=precision_recall_fscore_support(Groundt_all,Prediction_all)
    print('Cats Precision:{}'.format(np.round(precision[0],3)),'   Dogs Precision:{}'.format(np.round(precision[1],3)))
    print('Cats Recall:{}'.format(np.round(recall[0],3)),'   Dogs Recall:{}'.format(np.round(recall[1],3)))
    print('Cats Fscore:{}'.format(np.round(fbeta_score[0],3)),'   Dogs Fscore:{}'.format(np.round(fbeta_score[1],3)))
    
def main():

    args = parser.parse_args()

    train_sampler=None
    multiprocessing_distributed=True


    if args.resume!='':
        weight_path=args.resume
        resume=True
    else:
        resume=False    

    if args.only_validation!='':
        weight_path=args.only_validation
        only_validation=True
    else:
        only_validation=False 


    if torch.cuda.is_available():
        gpu=0
    else:
        gpu=None

    if not only_validation:
        train_dataset = datasets.ImageFolder(args.train_dir,transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                                                           transforms.RandomHorizontalFlip(),
                                                                           transforms.ToTensor()]))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.test_dir, transforms.Compose([transforms.Resize(args.image_size),transforms.CenterCrop(args.image_size),
                                                                      transforms.ToTensor()])),batch_size=args.batch_size, shuffle=False,
                                                                    num_workers=args.num_workers, pin_memory=True)

    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2);
    if resume or only_validation:
        if gpu is None:
            checkpoint = torch.load(weight_path,map_location='cpu')
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(weight_path, map_location=loc);
        best_acc1 = checkpoint['best_acc1']
        if type(best_acc1)!=torch.Tensor:
            best_acc1=torch.tensor(best_acc1)
        if gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(gpu)
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        model.load_state_dict(checkpoint['state_dict']);
    else:
        if gpu is not None:
            model = model.cuda(gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if gpu is not None:
        criterion=criterion.cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    if args.seed_n is not None:
        random.seed(args.seed_n)
        torch.manual_seed(args.seed_n)
        cudnn.deterministic = True
    ngpus_per_node = torch.cuda.device_count()

    if only_validation:
        acc1,Groundt_all,Prediction_all = validate(val_loader, model, criterion, gpu=gpu)
        print('Validation Average Accuracy:',acc1)
        calculate_precision_recall_fscore(Groundt_all,Prediction_all)
    else:
        best_acc1 = 0
        for epoch in range(0, args.epochs):
            adjust_learning_rate(optimizer, epoch, lr=args.lr)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, gpu=gpu)

            # evaluate on validation set
            acc1,Groundt_all,Prediction_all = validate(val_loader, model, criterion, gpu=gpu)
            print('Validation Average Accuracy:',acc1)
            calculate_precision_recall_fscore(Groundt_all,Prediction_all)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if (multiprocessing_distributed and -1 % ngpus_per_node == 0):
                save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),
                                 'best_acc1': best_acc1,'optimizer' : optimizer.state_dict(),}, is_best)
        acc1,Groundt_all,Prediction_all = validate(val_loader, model, criterion, gpu=gpu)
        print('Overall Average Accuracy:',acc1)
        calculate_precision_recall_fscore(Groundt_all,Prediction_all)
if __name__ == '__main__':
    main()
