import torch
import random
import re #split使用
import  numpy as np
from zhconv import convert #简繁转换


# 简繁转换 并构建词汇表
def build_word_dict(train_path):
    words = []
    max_len = 0
    total_len = 0
    with open(train_path,'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for line in  lines:
            line = convert(line, 'zh-cn') #转换成大陆简体
            line_words = re.split(r'[\s]', line)[1:-1] # 按照空字符\t\n 空格来切分
            max_len = max(max_len, len(line_words))
            total_len += len(line_words)
            for w in line_words:
                words.append(w)
    words = list(set(words))#最终去重
    words = sorted(words) # 一定要排序不然每次读取后生成此表都不一致，主要是set后顺序不同
    #用unknown来表示不在训练语料中的词汇
    word2ix = {w:i+1 for i,w in enumerate(words)} # 第0是unknown的 所以i+1
    ix2word = {i+1:w for i,w in enumerate(words)}
    word2ix['<unk>'] = 0
    ix2word[0] = '<unk>'
    avg_len = total_len / len(lines)
    return word2ix, ix2word, max_len,  avg_len

#准确率指标
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


## topk的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    #     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn

#混淆矩阵指标
class ConfuseMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        # 标签的分类：0 pos 1 neg
        self.confuse_mat = torch.zeros(2,2)
        self.tp = self.confuse_mat[0,0]
        self.fp = self.confuse_mat[0,1]
        self.tn = self.confuse_mat[1,1]
        self.fn = self.confuse_mat[1,0]
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.F1 = 0
    def update(self, output, label):
        pred = output.argmax(dim = 1)
        for l, p in zip(label.view(-1),pred.view(-1)):
            self.confuse_mat[p.long(), l.long()] += 1 # 对应的格子加1
        self.tp = self.confuse_mat[0,0]
        self.fp = self.confuse_mat[0,1]
        self.tn = self.confuse_mat[1,1]
        self.fn = self.confuse_mat[1,0]
        self.acc = (self.tp+self.tn) / self.confuse_mat.sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.F1 = 2 * self.pre*self.rec / (self.pre + self.rec)


def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  #并行gpu
        torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
        torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速

if __name__ == '__main__':
    word2ix, ix2word, max_len, avg_len = build_word_dict('D:/AIdata/Sentiment-classification/Dataset/train.txt')
    # print(word2ix)
    # print(ix2word)