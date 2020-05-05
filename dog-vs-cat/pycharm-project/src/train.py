from torch import nn
from dataset import MyDataset,dataset_split
from config import config as C
from model import MyCNN
import torch.optim as optim
from utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

def dataloader(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               shuffle=True, pin_memory=True, num_workers=0)
    return data_loader

def train( epochs, train_loader, device, model, criterion, optimizer,tensorboard_path):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        top1 = AvgrageMeter()
        train_loader = tqdm(train_loader)
        train_loss = 0.0
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', 0.001))
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()

    print('Finished Training')

def validate(validate_loader, device, model, criterion):
    val_acc = 0.0
    model = model.to(device)
    model.eval()
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        val_top1 = AvgrageMeter()
        validate_loader = tqdm(validate_loader)
        validate_loss = 0.0
        for i, data in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            #         inputs,labels = data[0],data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc

def submission(csv_path,test_loader, device, model):
    result_list = []
    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            softmax_func = nn.Softmax(dim=1)  # dim=1表示行的和为1
            soft_output = softmax_func(outputs)
            predicted = soft_output[:, 1]
            for i in range(len(predicted)):
                result_list.append({
                    "id": labels[i].item(),
                    "label": predicted[i].item()
                })
    columns = result_list[0].keys()
    result_dict = {col: [anno[col] for anno in result_list] for col in columns}
    result_df = pd.DataFrame(result_dict)
    result_df = result_df.sort_values("id")
    result_df.to_csv(csv_path, index=None)

def debug():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 2
    net = MyCNN()
    train_path = C.train_path
    test_path = C.test_path

    train_ds = MyDataset(train_path)
    new_train_ds, validate_ds = dataset_split(train_ds, 0.8)
    test_ds = MyDataset(test_path, train=False)

    train_loader = dataloader(train_ds)
    new_train_loader = dataloader(new_train_ds)
    validate_loader = dataloader(validate_ds)
    test_loader = dataloader(test_ds)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(epochs,new_train_loader,device,net,criterion,optimizer)
    print("validate acc:",validate(validate_loader,device,net,criterion))
    submission(csv_path=C.csv_path,test_loader=test_loader,device=device,model=net)
    torch.save(net.state_dict(), C.model_save_path)



if __name__ == '__main__':
    debug()
