import config
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import CommentDataSet, mycollate_fn
from model import SentimentModel, pre_weight
from utils import build_word_dict, set_seed
from train import *


def main():
    Config = config.get_args()
    set_seed(Config.seed)
    word2ix, ix2word, max_len, avg_len = build_word_dict(Config.train_path)

    test_data = CommentDataSet(Config.test_path, word2ix, ix2word)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False,
                             num_workers=0, collate_fn=mycollate_fn,)

    weight = torch.zeros(len(word2ix), Config.embedding_dim)

    model = SentimentModel(embedding_dim=Config.embedding_dim,
                           hidden_dim=Config.hidden_dim,
                           LSTM_layers=Config.LSTM_layers,
                           drop_prob=Config.drop_prob,
                           pre_weight=weight)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(
        torch.load(
            Config.model_save_path),
        strict=True)  # 模型加载

    confuse_meter = ConfuseMeter()
    confuse_meter = test(test_loader, device, model, criterion)



if __name__ == '__main__':
    main()
