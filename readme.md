# TextCNN


## 简介

TextCNN网络的实现，并且在MR数据集中实现训练与测试

论文地址为:

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)



## 环境


  * Pytorch = 1.5.0
  * GTX1080

## 代码说明

1. MR数据集下载地址：https://www.cs.cornell.edu/people/pabo/movie-review-data/
2. Word2vec下载地址: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 下载后放入根目录下的models文件夹中
3. 正则化方法:根据论文中的方法1)Dropout.2)L2正则
4. 采样10折交叉验证，提升模型的准确率
5. 可根据需要在utils中设置earlystopping
6. 数据集划分: 10%的Test数据集，90%的train,val数据集
7. 模型的详细参数均可以在config类中设置



## 后续工作:

1）融合到大规模预训练模型输出端，实现文本分类。

2）。。。