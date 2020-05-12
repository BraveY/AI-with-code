import argparse


def get_args():
    parser = argparse.ArgumentParser("Sentiment-calssfication")
    parser.add_argument(
        '--train_path',
        type=str,
        default='D:/AIdata/Sentiment-classification/Dataset/train.txt',
        required=False,
        help='train datasets  path')
    parser.add_argument(
        '--test_path',
        type=str,
        default='D:/AIdata/Sentiment-classification/Dataset/test.txt',
        required=False,
        help='test datasets  path')
    parser.add_argument(
        '--validation_path',
        type=str,
        default='D:/AIdata/Sentiment-classification/Dataset/validation.txt',
        required=False,
        help='validation datasets  path')
    parser.add_argument(
        '--pred_word2vec_path',
        type=str,
        default='D:/AIdata/Sentiment-classification/Dataset/wiki_word2vec_50.bin',
        required=False,
        help='pretrained word2vec path')
    parser.add_argument(
        '--tensorboard_path',
        type=str,
        default='./tensorboard',
        required=False,
        help='tensorboard file save path')
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='./modelDict/model.pth',
        required=False,
        help='model save path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=50,
        help='embedding dimension 50 to fit pretrain word2vec')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=100,
        help='lstm layer hidden state dimension')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size')
    parser.add_argument(
        '--LSTM_layers',
        type=int,
        default=3,
        help='layers num of LSTM')
    parser.add_argument(
        '--drop_prob',
        type=float,
        default=0.5,
        help='dropout probability')
    parser.add_argument('--epochs', type=int, default=3, help='batch size')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate')
    parser.add_argument('--comment_str', type=str, default='电影不错',
                        required=False, help='comment string ')
    args = parser.parse_args()
    return args



#     'lr':0.001,
#     'LSTM_layers':3,
#     'drop_prob': 0.5,
#     'seed':0
# })# class DictObj(object):
# #     # 私有变量是map
# #     # 设置变量的时候 初始化设置map
# #     def __init__(self, mp):
# #         self.map = mp
# #         # print(mp)
# #
# # # set 可以省略 如果直接初始化设置
# #     def __setattr__(self, name, value):
# #         if name == 'map':# 初始化的设置 走默认的方法
# #             # print("init set attr", name ,"value:", value)
# #             object.__setattr__(self, name, value)
# #             return
# #         # print('set attr called ', name, value)
# #         self.map[name] = value
# # # 之所以自己新建一个类就是为了能够实现直接调用名字的功能,同时可以修改配置。
# #     def __getattr__(self, name):
# #         # print('get attr called ', name)
# #         return  self.map[name]
#
#
# # Config = DictObj({
# #     'train_path' : "D:/AIdata/Sentiment-classification/Dataset/train.txt",
# #     'test_path' : "D:/AIdata/Sentiment-classification/Dataset/test.txt",
# #     'validation_path' : "D:/AIdata/Sentiment-classification/Dataset/validation.txt",
# #     'pred_word2vec_path':'D:/AIdata/Sentiment-classification/Dataset/wiki_word2vec_50.bin',
# #     'tensorboard_path':'./tensorboard',
# #     'model_save_path':'./modelDict/model.pth',
# #     'embedding_dim':50,
# #     'hidden_dim':100,
# args = get_args()
# Config = DictObj({
#     arg:getattr(args, arg) for arg in vars(args)
# })
