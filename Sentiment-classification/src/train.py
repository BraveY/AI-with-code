import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AvgrageMeter, ConfuseMeter, accuracy

# 一个epoch的训练逻辑


def train(
        epoch,
        epochs,
        train_loader,
        device,
        model,
        criterion,
        optimizer,
        scheduler,
        tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
        inputs, labels, batch_seq_len = data[0].to(
            device), data[1].to(device), data[2]
        # 初始为0，清除上个batch的梯度信息
        optimizer.zero_grad()
        outputs, hidden = model(inputs, batch_seq_len)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, pred = outputs.topk(1)
        prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (
            train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
        train_loader.set_postfix(log=postfix)

        # ternsorboard 曲线绘制
        if os.path.exists(tensorboard_path) == False:
            os.mkdir(tensorboard_path)
        writer = SummaryWriter(tensorboard_path)
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_scalar('Train/Accuracy', top1.avg, epoch)
        writer.flush()
    scheduler.step()

#     print('Finished Training')


def validate(
        epoch,
        validate_loader,
        device,
        model,
        criterion,
        tensorboard_path):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = tqdm(validate_loader)
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(
                device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs, _ = model(inputs, batch_seq_len)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (
                validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            if os.path.exists(tensorboard_path) == False:
                os.mkdir(tensorboard_path)
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Validate/Loss', loss.item(), epoch)
            writer.add_scalar('Validate/Accuracy', val_top1.avg, epoch)
            writer.flush()
        val_acc = val_top1.avg
    return val_acc


def test(validate_loader, device, model, criterion):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    confuse_meter = ConfuseMeter()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = tqdm(validate_loader)
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels, batch_seq_len = data[0].to(
                device), data[1].to(device), data[2]
            #         inputs,labels = data[0],data[1]
            outputs, _ = model(inputs, batch_seq_len)
#             loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            confuse_meter.update(outputs, labels)
#             validate_loss += loss.item()
            postfix = {'test_acc': '%.6f' % val_top1.avg,
                       'confuse_acc': '%.6f' % confuse_meter.acc}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return confuse_meter
