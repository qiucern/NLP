import torch
from tqdm import tqdm
# ⚠️ 踢掉 sklearn，换上大名鼎鼎的 seqeval
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, dataloader, device, id2label):
    """
    注意：传入了一个 id2label 字典，用于把数字 0, 1, 2 转回 'O', 'B-Aspect', 'I-Aspect'
    """
    model.eval()
    all_preds_str = []
    all_trues_str = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            # CRF 解码预测
            batch_preds = model(input_ids, attention_mask, adj_matrix, labels=None)
            
            for i in range(len(batch_preds)):
                pred_path = batch_preds[i] 
                # 截断填充，拿到正确的数字序列
                true_path = labels[i][:len(pred_path)].cpu().numpy().tolist() 
                
                # ⚠️ 核心改变：不拉平列表！直接把数字映射回字符串标签！
                # 假设 pred_path 是 [0, 1, 2, 0], id2label 转换后变成 ['O', 'B-Aspect', 'I-Aspect', 'O']
                pred_str = [id2label[p] for p in pred_path]
                true_str = [id2label[t] for t in true_path]
                
                # 保持列表的列表（List of Lists）结构
                all_preds_str.append(pred_str)
                all_trues_str.append(true_str)
                
    # --- 调用 seqeval 计算严苛的实体级指标 ---
    acc = accuracy_score(all_trues_str, all_preds_str)
    # seqeval 默认就是针对实体跨度（Span）进行计算的
    p = precision_score(all_trues_str, all_preds_str)
    r = recall_score(all_trues_str, all_preds_str)
    f1 = f1_score(all_trues_str, all_preds_str)
    # 在 evaluate.py 中，对比 pred_str 和 true_str
    if pred_str != true_str:
        # 只打印前 5 个看一看
        # print(f"原句 Token: {tokenizer.convert_ids_to_tokens(input_ids[i])}")
        print(f"真实标签: {true_str}")
        print(f"模型预测: {pred_str}")
        print("-" * 30)
    
    return p, r, f1, acc