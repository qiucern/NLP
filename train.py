import json
import os
from skimage import data_dir
import torch
from torch.optim import AdamW
from tqdm import tqdm 
from models.main_model import ABSAMainModel
from evaluate import evaluate_model
from data_loader.dataset import ABSADataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils.logger import log_experiment_to_csv
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="ABSA GCN-CRF 训练流水线")
    
    # 定义允许在外部修改的超参数，并赋予默认值（即你之前的黄金组合）
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='RoBERTa 基准学习率')
    parser.add_argument('--gcn_dim', type=int, default=300, help='GCN 的输出维度 out_features')
    parser.add_argument('--num_layers', type=int, default=2, help='GCN 的层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='GCN 内部的 Dropout 比例')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    
    args = parser.parse_args()

    # 将解析出来的动态参数，赋值给训练流程（后续代码完全不需要改动）
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    GCN_DIM = args.gcn_dim
    NUM_LAYERS = args.num_layers
    DROPOUT = args.dropout
    SEED = args.seed

    MAX_LEN = 128
    NUM_TAGS = 7 
    TOKENIZER_NAME = 'roberta-base'
    # # 1. 超参数配置
    # EPOCHS = 40
    # BATCH_SIZE = 32
    # LEARNING_RATE = 2e-5 
    # MAX_LEN = 128
    # NUM_TAGS = 7 
    # TOKENIZER_NAME = 'roberta-base' 
    # SEED = 42 # 将 Seed 提升为全局配置
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_JSON = os.path.join(BASE_DIR, "data", "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [INIT] 设备: {device}")

    print("\n📦 正在加载静态数据集...")
    train_ds = ABSADataset(os.path.join(DATA_JSON,"train.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)
    val_ds = ABSADataset(os.path.join(DATA_JSON,"val.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)
    test_ds = ABSADataset(os.path.join(DATA_JSON,"test.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 实例化模型 ---
    model = ABSAMainModel(model_name_or_path=TOKENIZER_NAME, num_tags=NUM_TAGS, gcn_out_dim=GCN_DIM).to(device)
    
    # 🌟 新增：参数分组与分层学习率
    roberta_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        # 如果参数名字里包含 'roberta'（具体看你模型里预训练层的命名，通常是这个）
        if "roberta" in name or "encoder" in name:
            roberta_params.append(param)
        else:
            # 剩下的 GCN、Attention、CRF 全部分到头部参数里
            head_params.append(param)
            
    # 🌟 组合进优化器，给头部赋予 100 倍的极速学习率
    optimizer = AdamW([
        {'params': roberta_params, 'lr': LEARNING_RATE},          # 2e-5
        {'params': head_params, 'lr': LEARNING_RATE * 100}        # 2e-3
    ])
    
    # 后面的 Warmup 调度器代码保持不变...
    total_steps = len(train_loader) * EPOCHS
    # ...

    # 🌟 新增：计算总训练步数
    total_steps = len(train_loader) * EPOCHS
    # 🌟 新增：设置预热步数（通常占总步数的 10%）
    warmup_steps = int(total_steps * 0.1)

    # 🌟 新增：定义动态学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    best_val_f1 = 0.0 

    # ⚠️ 必须补全 7 个标签，否则 seqeval 解码时会报 KeyError
    id2label = {
        0: "O",
        1: "B-Aspect",
        2: "I-Aspect",
        3: "B-Positive",  # 假设的极性标签，请按你的真实数据修改
        4: "I-Positive",
        5: "B-Negative",
        6: "I-Negative"
    }

    # 4. 训练大循环
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} Epoch {epoch}/{EPOCHS} {'='*20}")
        
        # --- 4.1 核心训练逻辑 ---
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, adj_matrix, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step() # 🌟 让学习率按计划平滑衰减
            optimizer.zero_grad() # 习惯上 zero_grad 放在 step 之后也可以
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # ⚠️ 缩进修复：以下代码必须在 for epoch 循环内部执行！
        print(f"🔥 [TRAIN] 平均 Loss: {total_loss/len(train_loader):.4f}")
            
        # --- 4.2 调用独立的评估模块 (每个 Epoch 跑一次) ---
        val_p, val_r, val_f1, val_acc = evaluate_model(model, val_loader, device, id2label)
        
        print(f"📊 [VALIDATION] Acc: {val_acc:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"🌟 新纪录！当前最佳 F1: {best_val_f1:.4f} (Acc: {val_acc:.4f})")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🎉 验证集创新高！权重已保存至: {BEST_MODEL_PATH}")
            
    # 5. 最终盲测 (跳出 Epoch 循环后执行)
    print("\n" + "*"*40)
    print("🏆 开始测试集最终盲测...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_p, test_r, test_f1, test_acc = evaluate_model(model, test_loader, device, id2label)
    
    print(f"✅ 终极无偏测试结果 (Span-level):")
    print(f"   - 准确率 (Accuracy) : {test_acc:.4f}")
    print(f"   - 精确率 (Precision): {test_p:.4f}")
    print(f"   - 召回率 (Recall)   : {test_r:.4f}")
    print(f"   - 宏平均 F1 Score   : {test_f1:.4f}")

    # ...... 前面是测试集的最终盲测打印代码 ......
    
    # ==========================================
    # 🌟 规范调用：记录自动化实验台账
    # ==========================================
    # 1. 整理你本次修改过的超参数
    args_dict = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Epochs": EPOCHS,
        "BatchSize": BATCH_SIZE,
        "RoBERTa_LR": LEARNING_RATE,
        "Head_LR": LEARNING_RATE * 100,
        "GCN_Dim": GCN_DIM,
        "Num_Layers": NUM_LAYERS  # 记录下 GCN 的层数
    }
    
    # 2. 整理模型跑出来的最终成绩
    metrics_dict = {
        "Best_Val_F1": f"{best_val_f1:.4f}",
        "Test_F1": f"{test_f1:.4f}",
        "Test_Precision": f"{test_p:.4f}",
        "Test_Recall": f"{test_r:.4f}"
    }
    
    # 3. 优雅地把字典交给 logger 处理
    csv_path = os.path.join(BASE_DIR, "experiment_results.csv")
    log_experiment_to_csv(csv_path, args_dict, metrics_dict)
    print(f"\n📁 规范化入账成功！本次实验结果已归档至: {csv_path}")

if __name__ == "__main__":
    main()