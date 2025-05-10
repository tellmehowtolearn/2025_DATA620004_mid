
# 微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

## 实验简介
本实验旨在比较使用预训练模型（如 ImageNet 上预训练的 ResNet18/AlexNet）与从头开始随机初始化模型在 Caltech-101 数据集上的表现差异。

## 运行步骤

### 1. 下载数据集以及预处理
预先下载好数据集[Caltech101](https://data.caltech.edu/records/mzrjq-6wc02)并解压到当前目录，删除目录`101_ObjectCategories`下的`BACKGROUND_Google`文件夹。然后运行`dataset_split.py`脚本进行数据集划分，划分为训练集和验证集。该脚本会在当前目录下创建一个`caltech101_split`文件夹，里面包含了训练集和验证集以及测试集的三个子文件夹。

### 2. 超参选择
通过运行`param_grid.py`脚本来选择超参数。该脚本会针对每种模型类型自动生成一个包含超参数组合的 CSV 文件，从而得到最优的超参数组合。对于预训练模型和随机初始化模型，需要手动修改参数`init_type`。


### 3. 训练模型
根据超参选择得到的最优超参数组合，运行`train.py`脚本进行模型训练。以下是一个示例命令：
```bash
python train.py --model_type resnet18 --init_type pretrained --batch_size 32 --epochs 10 --lr_fc 1e-3 --lr_backbone 1e-4
```

### 4. 启动 TensorBoard 查看训练过程
```bash
tensorboard --logdir=runs
```

### 5. 测试模型
运行`test.py`脚本进行模型测试。以下是一个示例命令：
```bash
python test.py --model_type resnet18 --checkpoint "./checkpoints/resnet18_pretrained_best.pth" --data_dir "caltech101_split" --batch_size 32
```

浏览器访问 http://localhost:6006 查看训练损失与验证准确率变化。

## 注意事项
- 对于`train.py`若使用 `--tune_mode` 参数，则模型不会保存，但训练日志仍会保留。
- 日志记录与 TensorBoard 可视化自动写入 `runs/` 和 `logs/` 文件夹。
- 训练完成后，模型会保存在 `checkpoints/` 文件夹中。
- 其他具体参数设置请参考代码中的注释，其他细节参看报告文档。