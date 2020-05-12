import torch
import os
import config
import shutil
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import CommentDataSet, mycollate_fn
from model import SentimentModel, pre_weight
from utils import build_word_dict, set_seed
from train import *


def main():
    Config = config.get_args()
    set_seed(Config.seed)
    word2ix, ix2word, max_len, avg_len = build_word_dict(Config.train_path)

    train_data = CommentDataSet(Config.train_path, word2ix, ix2word)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                              num_workers=0, collate_fn=mycollate_fn,)
    validation_data = CommentDataSet(Config.validation_path, word2ix, ix2word)
    validation_loader = DataLoader(
        validation_data,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=mycollate_fn,
    )
    test_data = CommentDataSet(Config.test_path, word2ix, ix2word)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False,
                             num_workers=0, collate_fn=mycollate_fn,)

    weight = pre_weight(
        len(word2ix),
        Config.pred_word2vec_path,
        Config.embedding_dim,
        word2ix,
        ix2word)

    model = SentimentModel(embedding_dim=Config.embedding_dim,
                           hidden_dim=Config.hidden_dim,
                           LSTM_layers=Config.LSTM_layers,
                           drop_prob=Config.drop_prob,
                           pre_weight=weight)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)  # 学习率调整
    criterion = nn.CrossEntropyLoss()

    # 因为使用tensorboard画图会产生很多日志文件，这里进行清空操作

    if os.path.exists(Config.tensorboard_path):
        shutil.rmtree(Config.tensorboard_path)
        os.mkdir(Config.tensorboard_path)

    for epoch in range(Config.epochs):
        train_loader = tqdm(train_loader)
        train_loader.set_description(
            '[%s%04d/%04d %s%f]' %
            ('Epoch:', epoch + 1, Config.epochs, 'lr:', scheduler.get_lr()[0]))
        train(
            epoch,
            Config.epochs,
            train_loader,
            device,
            model,
            criterion,
            optimizer,
            scheduler,
            Config.tensorboard_path)
        validate(
            epoch,
            validation_loader,
            device,
            model,
            criterion,
            Config.tensorboard_path)

    # 模型保存
    if os.path.exists(Config.model_save_path) == False:
        os.mkdir('./modelDict/')
    torch.save(model.state_dict(), Config.model_save_path)

    confuse_meter = ConfuseMeter()
    confuse_meter = test(test_loader, device, model, criterion)


if __name__ == '__main__':
    main()
