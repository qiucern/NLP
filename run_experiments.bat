@echo off
:: 设置字符集为 UTF-8，防止 Windows 终端打印中文乱码
chcp 65001
cls


echo 基线模型 (Dim=300, Layers=3, Dropout=0.1)...
python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 3 --dropout 0.1

:: --------------------------------------------------------
:: 增加 GCN 层
:: --------------------------------------------------------
echo 更改图层(Dim=300, Layers=5, Dropout=0.1)...
python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 5 --dropout 0.1

:: --------------------------------------------------------
:: 减少 GCN 层
:: --------------------------------------------------------
echo 正在启动第四组：轻量化图层打破过平滑 (Dim=300, Layers=1, Dropout=0.1)...
python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 1 --dropout 0.1

:: --------------------------------------------------------
:: 放大维度
:: --------------------------------------------------------
echo 正在启动第二组：扩容特征维度 (Dim=512, Layers=2, Dropout=0.1)...
python train.py --epochs 100 --batch_size 32 --gcn_dim 768 --num_layers 3 --dropout 0.1

:: --------------------------------------------------------
:: 降低Dropout
:: --------------------------------------------------------
echo 正在启动第三组：维度放大的高正则化防过拟合 (Dim=300, Layers=2, Dropout=0.0001)...
python train.py --epochs 100 --batch_size 32 --gcn_dim 300 --num_layers 3 --dropout 0.0001
pause