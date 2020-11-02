import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    
    def __init__(self, config):
        super(TextCNN, self).__init__()

        # 词嵌入层
        if config.use_pretrained_embed:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)  # 加载词向量模型,可以继续训练
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        #卷积层
        if config.use_cuda and torch.cuda.is_available():
            self.convs = [
                nn.Sequential(
                    nn.Conv1d(config.embedding_dim, config.filters_number, filter_size),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=config.max_seq_len - filter_size + 1)
                ).cuda()
                for filter_size in config.filters]
        else:
            self.convs = [
                nn.Sequential(
                    nn.Conv1d(config.embedding_dim, config.filters_number, filter_size),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=config.max_seq_len - filter_size + 1)
                )
                for filter_size in config.filters]

        # # 卷积层
        # if config.use_cuda:
        #     self.convs = [nn.Conv1d(config.embedding_dim, config.filters_number, filter_size).cuda()
        #                   for filter_size in config.filters]
        # else:
        #     self.convs = [nn.Conv1d(config.embedding_dim, config.filters_number, filter_size)
        #                   for filter_size in config.filters]

        # 正则化处理
        self.dropout = nn.Dropout(config.dropout, inplace=True)

        # 分类层
        self.fc = nn.Linear(config.filters_number*len(config.filters), config.label_num)

    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x))
        # 池化层
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2).contiguous()
        x = torch.cat([conv_relu_pool(x) for conv_relu_pool in self.convs], dim=1).squeeze(2)
        # x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x