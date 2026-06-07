import torch
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

# 🌟 新增参数：tokenizer 和 is_test
def evaluate_model(model, dataloader, device, id2label, tokenizer=None, is_test=False):
    model.eval()
    all_preds_str = []
    all_trues_str = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            batch_preds = model(input_ids, attention_mask, adj_matrix, labels=None)
            
            for i in range(len(batch_preds)):
                pred_path = batch_preds[i] 
                true_path = labels[i][:len(pred_path)].cpu().numpy().tolist() 
                
                pred_str = [id2label.get(p, "O") for p in pred_path]
                true_str = [id2label.get(t, "O") for t in true_path]
                
                all_preds_str.append(pred_str)
                all_trues_str.append(true_str)
                
                # 🌟 修复后的核心逻辑：在测试集阶段，发现预测错误，立刻记入文本
                if is_test and pred_str != true_str and tokenizer is not None:
                    # 获取原句汉字，去掉无用的占位符
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                    clean_tokens = [t for t in tokens if t not in ['<pad>', '<s>', '</s>', '[PAD]', '[CLS]', '[SEP]']]
                    
                    with open("bad_cases_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"【原句】: {' '.join(clean_tokens)}\n")
                        f.write(f"【真实】: {true_str}\n")
                        f.write(f"【预测】: {pred_str}\n")
                        f.write("-" * 50 + "\n")
                
    acc = accuracy_score(all_trues_str, all_preds_str)
    p = precision_score(all_trues_str, all_preds_str, zero_division=0)
    r = recall_score(all_trues_str, all_preds_str, zero_division=0)
    f1 = f1_score(all_trues_str, all_preds_str, zero_division=0)
    
    return p, r, f1, acc