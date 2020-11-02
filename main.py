import os
import torch
from config.config import Config
from tools.data_preprocessing import MR_dataset
from torch.utils.data import DataLoader
from textcnn import TextCNN
from tools.utils import EarlyStopping
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import torch.nn.functional as F


# Create the configuration
def get_test_result(model, data_iter, data_set, config, criterion):
    # 生成测试结果
    model.eval()
    data_loss = 0
    true_sample_num = 0
    for data, label in data_iter:
        if config.use_cuda and torch.cuda.is_available():
            data = data.to(torch.int64).cuda()
            label = label.cuda()
        out = model(data)
        loss = criterion(out, autograd.Variable(label.long()))
        data_loss += loss.data.item()
        # print(out)
        true_sample_num += np.sum((torch.argmax(F.softmax(out, dim=1), 1) == label).cpu().numpy())
    acc = true_sample_num / len(data_set)
    return data_loss, acc


def main():
    for i in range(10):
        # 加载配置文件
        config = Config()
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        # 加载数据集
        early_stopping = EarlyStopping(patience=10, verbose=True, cv_index=i)
        kwargs = {'num_workers': 2, 'pin_memory': True}
        dataset_train = MR_dataset(config=config, state="train", k=i, embedding_state=True)
        train_data_batch = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=False, drop_last=False, **kwargs)
        dataset_valid = MR_dataset(config=config, state="valid", k=i, embedding_state=False)
        valid_data_batch = DataLoader(dataset_valid, batch_size=config.batch_size, shuffle=False, drop_last=False, **kwargs)
        dataset_test = MR_dataset(config=config, state="test", k=i, embedding_state=False)
        test_data_batch = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, drop_last=False, **kwargs)
        print(len(dataset_train), len(dataset_valid), len(dataset_test))

        if config.use_pretrained_embed:
            config.embedding_pretrained = torch.from_numpy(dataset_train.weight).float().cuda()
            print("load pretrained models.")
        else:
            config.embedding_pretrained = None

        config.vocab_size = dataset_train.vocab_size

        model = TextCNN(config)
        print(model)

        if config.use_cuda and torch.cuda.is_available():
            # print("load data to CUDA")
            model.cuda()
            # config.embedding_pretrained.cuda()

        criterion = nn.CrossEntropyLoss()       # 定义为交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        count = 0
        loss_sum = 0.0
        for epoch in range(config.epoch):
            # 开始训练
            model.train()
            for data, label in train_data_batch:
                if config.use_cuda and torch.cuda.is_available():
                    data = data.to(torch.int64).cuda()
                    label = label.cuda()
                else:
                    data.to(torch.int64)
                # data = torch.autograd.Variable(data).long().cuda()
                # label = torch.autograd.Variable(label).squeeze()
                out = model(data)
                l2_loss = config.l2_weight * torch.sum(torch.pow(list(model.parameters())[1], 2))
                loss = criterion(out, autograd.Variable(label.long())) + l2_loss
                loss_sum += loss.data.item()
                count += 1
                if count % 100 == 0:
                    print("epoch", epoch, end='  ')
                    print("The loss is: %.5f" % (loss_sum / 100))
                    loss_sum = 0
                    count = 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 一轮训练结束，在验证集测试
            valid_loss, valid_acc = get_test_result(model, valid_data_batch, dataset_valid, config, criterion)
            early_stopping(valid_loss, model, config)
            print("The valid acc is: %.5f" % valid_acc)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # 1 fold训练结果
        model.load_state_dict(torch.load(os.path.abspath(os.path.join(config.checkpoint_path, 'checkpoint%d.pt'% i))))
        test_loss, test_acc = get_test_result(model, test_data_batch, dataset_test, config, criterion)
        print("The test acc is: %.5f" % test_acc)


if __name__ == "__main__":
    main()
