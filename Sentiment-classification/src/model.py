import torch
import gensim # word2vec预训练加载
from torch import nn
# 变长序列的处理
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence




def pre_weight(vocab_size,pred_word2vec_path, embedding_dim,word2ix, ix2word):
    # word2vec加载
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(pred_word2vec_path, binary=True)
    weight = torch.zeros(vocab_size,embedding_dim)
    #初始权重
    for i in range(len(word2vec_model.index2word)):#预训练中没有word2ix，所以只能用索引来遍历
        try:
            index = word2ix[word2vec_model.index2word[i]]#得到预训练中的词汇的新索引
        except:
            continue
        weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
            ix2word[word2ix[word2vec_model.index2word[i]]]))#得到对应的词向量
    return weight


class SentimentModel(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim,
                 LSTM_layers,drop_prob,pre_weight):
        super(SentimentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.LSTM_layers = LSTM_layers
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True, dropout=drop_prob, bidirectional=False)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    #         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, batch_seq_len, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        embeds = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(self.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))  # hidden 是h,和c 这两个隐状态
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout(torch.tanh(self.fc1(output)))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        last_outputs = self.get_last_output(output, batch_seq_len)
        #         output = output.reshape(batch_size * seq_len, -1)
        return last_outputs, hidden

    def get_last_output(self, output, batch_seq_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(len(batch_seq_len)):
            last_outputs[i] = output[i][batch_seq_len[i] - 1]  # index 是长度 -1
        last_outputs = last_outputs.to(output.device)
        return last_outputs