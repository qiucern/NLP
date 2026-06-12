@echo off
:: 设置字符集为 UTF-8，防止 Windows 终端打印中文乱码
chcp 65001
cls


@REM echo 基线模型 (Dim=300, Layers=3, Dropout=0.1)...
@REM python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 3 --dropout 0.1

@REM :: --------------------------------------------------------
@REM :: 增加 GCN 层
@REM :: --------------------------------------------------------
@REM echo 更改图层(Dim=300, Layers=5, Dropout=0.1)...
@REM python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 5 --dropout 0.1

@REM :: --------------------------------------------------------
@REM :: 减少 GCN 层
@REM :: --------------------------------------------------------
@REM echo 正在启动第四组：轻量化图层打破过平滑 (Dim=300, Layers=1, Dropout=0.1)...
@REM python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 1 --dropout 0.1

@REM :: --------------------------------------------------------
@REM :: 放大维度
@REM :: --------------------------------------------------------
@REM echo 正在启动第二组：扩容特征维度 (Dim=512, Layers=2, Dropout=0.1)...
@REM python train.py --epochs 100 --batch_size 32 --gcn_dim 768 --num_layers 3 --dropout 0.1

@REM :: --------------------------------------------------------
@REM :: 降低Dropout
@REM :: --------------------------------------------------------
@REM echo 正在启动第三组：维度放大的高正则化防过拟合 (Dim=300, Layers=2, Dropout=0.0001)...
@REM python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 3 --dropout 0.0001
@REM pause


echo 正在训练基线模型 1：Vanilla RoBERTa + Linear...
python baseline_train_1.py --epochs 20 --batch_size 32 --dropout 0.1

echo 正在训练基线模型 2：Vanilla RoBERTa + CRF...
python baseline_train_2.py --epochs 20 --batch_size 32 --dropout 0.1

echo 正在训练基线模型 3：RoBERTa + BiLSTM + CRF...
python baseline_train_3.py --epochs 20 --batch_size 32 --dropout 0.1