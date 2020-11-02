import os
import random
import numpy as np
from torch.utils.data import Dataset
from gensim.models import KeyedVectors


class MR_dataset(Dataset):

    def __init__(self, config, state="train", k=3, embedding_state=True):
        super(MR_dataset, self).__init__()
        pos_samples = open(os.path.join(config.dataset_path, "rt-polarity.pos"), errors="ignore").readlines()  # 正样本数据
        neg_samples = open(os.path.join(config.dataset_path, "rt-polarity.neg"), errors="ignore").readlines()  # 负样本数据
        datas = pos_samples + neg_samples
        datas = [data.split() for data in datas]                        # 句子单词分割
        max_seq_len = max([len(data) for data in datas])                # 计算最长的句子,进行padding
        labels = [1] * len(pos_samples) + [0] * len(neg_samples)        # 正负样本标签
        word2id = {"<pad>": 0}
        datas_ids = []
        for i, data in enumerate(datas):
            data_ids = []
            for j, word in enumerate(data):
                if word not in word2id.keys():
                    word2id[word] = len(word2id)
                data_ids.append(word2id[word])
            data_ids.extend([0] * (max_seq_len - len(data)))
            assert len(data_ids) == max_seq_len             # 判断padding之后的句子长度是否正确
            datas_ids.append(data_ids)
        config.max_seq_len = max_seq_len
        self.vocab_size = len(word2id)          # 词向量大小
        self.word2id = word2id
        if embedding_state:
            self.weight = self.get_word2vec(config)
        else:
            self.weight = None

        self.embedding_dim = config.embedding_dim

        # 对数据集进性随机打乱
        c = list(zip(datas_ids, labels))
        random.seed(1)
        random.shuffle(c)
        datas_ids[:], labels[:] = zip(*c)
        if state == "train":        # 生成训练集
            self.datas_ids = datas_ids[:int(k * len(datas_ids) / 10)] + datas_ids[int((k + 1) * len(datas_ids) / 10):]
            self.labels = labels[:int(k * len(datas_ids) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas_ids = np.array(self.datas_ids[0:int(0.9*len(self.datas_ids))])
            self.labels = np.array(self.labels[0:int(0.9*len(self.labels))])
        elif state == "valid":      # 生成验证集
            self.datas_ids = datas_ids[:int(k * len(datas_ids) / 10)] + datas_ids[int((k + 1) * len(datas_ids) / 10):]
            self.labels = labels[:int(k * len(datas_ids) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas_ids = np.array(self.datas_ids[int(0.9 * len(self.datas_ids)):])
            self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
        elif state == "test":       # 生成测试集
            self.datas_ids = np.array(datas_ids[int(k * len(datas_ids) / 10):int((k + 1) * len(datas_ids) / 10)])
            self.labels = np.array(labels[int(k * len(labels) / 10):int((k + 1) * len(labels) / 10)])

    def __getitem__(self, item):
        return self.datas_ids[item], self.labels[item]

    def __len__(self):
        return len(self.datas_ids)

    def get_word2vec(self, config):
        # 加载训练好的词向量数据
        if not os.path.exists(os.path.abspath(os.path.join(config.pretrain_path, "word2vec_embedding_mr.npy"))):
            # 已经处理好的权重文件不存在,则开始处理预处理文件
            print("Reading word2vec Embedding...")
            word2vec_model = KeyedVectors.load_word2vec_format(os.path.abspath(os.path.join(config.pretrain_path, "GoogleNews-vectors-negative300.bin.gz")),binary=True)

            # 计算训练好的词向量的均值与方差,初始化权重矩阵时设置
            tmp = []
            oov_number = 0
            for word, index in self.word2id.items():
                try:
                    tmp.append(word2vec_model.get_vector(word=word))
                except Exception as Exp:
                    # print(Exp)
                    oov_number += 1
                    pass
            print("oov words number is ", oov_number)
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print("mean is {}.std is {}".format(mean, std))
            embedding_dim = self.embedding_dim        # word2vec 向量的维度  300
            embedding_weights = np.random.normal(mean, std, [self.vocab_size, embedding_dim])   # 随机化词向量矩阵
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = word2vec_model.get_vector(word)
                except:
                    pass
            np.save(os.path.abspath(os.path.join(config.pretrain_path, "word2vec_embedding_mr.npy")), embedding_weights)
        else:
            embedding_weights = np.load(os.path.abspath(os.path.join(config.pretrain_path, "word2vec_embedding_mr.npy")))

        return embedding_weights



