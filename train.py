import json
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm 
from models.main_model import ABSAMainModel
from evaluate import evaluate_model
from data_loader.dataset import ABSADataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from utils.logger import log_experiment_to_csv
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="ABSA GCN-CRF 训练流水线")
    
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='RoBERTa 基准学习率')
    parser.add_argument('--gcn_dim', type=int, default=300, help='GCN 的输出维度')
    parser.add_argument('--num_layers', type=int, default=1, help='GCN 的层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比例')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    
    # 🌟 消融实验
    parser.add_argument('--use_gcn', type=int, default=0, help='是否使用GCN (1:是, 0:否)')
    parser.add_argument('--use_attn', type=int, default=0, help='是否使用Attention (1:是, 0:否)')
    parser.add_argument('--use_real_adj', type=int, default=1, help='是否使用真实句法树 (1:是, 0:随机矩阵)')
    
    args = parser.parse_args()

    # 提取动态参数
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
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_JSON = os.path.join(BASE_DIR, "data", "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [INIT] 设备: {device}")

    print("\n📦 正在加载静态数据集...")
    train_ds = ABSADataset(os.path.join(DATA_JSON,"laptops_train.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)
    val_ds = ABSADataset(os.path.join(DATA_JSON,"laptops_val.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)
    test_ds = ABSADataset(os.path.join(DATA_JSON,"laptops_test.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # 实例化模型时，传入开关参数
    model = ABSAMainModel(
        model_name_or_path=TOKENIZER_NAME, 
        num_tags=NUM_TAGS, 
        gcn_out_dim=GCN_DIM,
        dropout_rate=DROPOUT,
        use_gcn=bool(args.use_gcn),
        use_attn=bool(args.use_attn)
    ).to(device)

    # --- 优化器与分层学习率配置 ---
    roberta_params = []
    gate_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "roberta" in name or "encoder" in name:
            roberta_params.append(param)
        elif "gate" in name:
            # 🌟 核心保护：把门控参数单独抓出来
            gate_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = AdamW([
        {'params': roberta_params, 'lr': LEARNING_RATE},          # 比如 2e-5
        {'params': gate_params,    'lr': LEARNING_RATE},          # 必须用基础极小学习率，绝对不能加倍！
        {'params': head_params,    'lr': LEARNING_RATE * 100}      # 顶层（GCN, Attn, CRF）用 10 倍即可
    ])
    
    # # --- 优化器与分层学习率配置 ---
    # roberta_params = []
    # head_params = []
    # for name, param in model.named_parameters():
    #     if "roberta" in name or "encoder" in name:
    #         roberta_params.append(param)
    #     else:
    #         head_params.append(param)
            
    # optimizer = AdamW([
    #     {'params': roberta_params, 'lr': LEARNING_RATE},          
    #     {'params': head_params, 'lr': LEARNING_RATE * 100}        
    # ])
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0 

    # 标签映射字典
    id2label = {
        0: "O",
        1: "B-Aspect",
        2: "I-Aspect",
        3: "B-Positive", 
        4: "I-Positive",
        5: "B-Negative",
        6: "I-Negative"
    }

    # ==================================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} Epoch {epoch}/{EPOCHS} {'='*20}")
        
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc="Training")

        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            # 🌟 随机矩阵处理：如果关闭真实句法树，直接生成一个随机噪音矩阵来干扰模型
            if not bool(args.use_real_adj):
                adj_matrix = torch.rand_like(adj_matrix).to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, adj_matrix, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"🔥 [TRAIN] 平均 Loss: {total_loss/len(train_loader):.4f}")
            
        # 验证集评估
        val_p, val_r, val_f1, val_acc = evaluate_model(model, val_loader, device, id2label)
        print(f"📊 [VALIDATION] Acc: {val_acc:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"🌟 新纪录！当前最佳 F1: {best_val_f1:.4f} (Acc: {val_acc:.4f})")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🎉 验证集创新高！权重已保存至: {BEST_MODEL_PATH}")
            
    print("\n" + "*"*40)
    print("🏆 开始测试集最终盲测...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # 🌟 传入 tokenizer，开启 is_test=True 触发错题本记录
    test_p, test_r, test_f1, test_acc = evaluate_model(
        model, test_loader, device, id2label, tokenizer=tokenizer, is_test=True
    )
    
    print(f"✅ 终极无偏测试结果 (Span-level):")
    print(f"   - 准确率 (Accuracy) : {test_acc:.4f}")
    print(f"   - 精确率 (Precision): {test_p:.4f}")
    print(f"   - 召回率 (Recall)   : {test_r:.4f}")
    print(f"   - 宏平均 F1 Score   : {test_f1:.4f}")
    
    args_dict = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Epochs": EPOCHS,
        "BatchSize": BATCH_SIZE,
        "RoBERTa_LR": LEARNING_RATE,
        "GCN_Dim": GCN_DIM,
        "GCN_Layers": NUM_LAYERS,
        "Dropout": DROPOUT,
        "use_GCN": args.use_gcn,
        "use_Attn": args.use_attn,
        "use_Real_Adj": args.use_real_adj
    }
    
    metrics_dict = {
        "Best_Val_F1": f"{best_val_f1:.4f}",
        "Test_F1": f"{test_f1:.4f}",
        "Test_Precision": f"{test_p:.4f}",
        "Test_Recall": f"{test_r:.4f}"
    }
    
    csv_path = os.path.join(BASE_DIR, "experiment_results_self.csv")
    log_experiment_to_csv(csv_path, args_dict, metrics_dict)
    print(f"实验结果记录至: {csv_path}")

if __name__ == "__main__":
    main()