import os


class Config(object):
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))   # 设置ROOT目录
        self.dataset_path = os.path.abspath(os.path.join(base_dir, "dataset/MR/"))
        self.pretrain_path = os.path.abspath(os.path.join(base_dir, "models/"))    # word2vec词向量文件
        self.use_pretrained_embed = True        # 是否使用预训练模型
        self.embedding_pretrained = None
        self.checkpoint_path = os.path.abspath(os.path.join(base_dir, "checkpoint/"))
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        # 训练参数
        self.batch_size = 64
        self.use_cuda = True
        self.epoch = 200
        self.learning_rate = 0.0005

        # 模型参数
        self.vocab_size = 30000         # 词典大小
        self.embedding_dim = 300        # 词向量维度
        self.filters = [2, 3, 4, 5]     # 卷积核大小设置
        self.filters_number = 100       # 卷积核个数
        self.max_seq_len = 59           # 序列最长长度
        self.dropout = 0.5              # dropout参数
        self.label_num = 2              # 分类类别
        self.l2_weight = 0.004          # l2正则超参数


if __name__ == "__main__":
    config = Config()
