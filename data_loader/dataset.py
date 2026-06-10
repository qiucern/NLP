# PyTorch Dataset 类
import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from data_loader.graph_builder import DependencyGraphBuilder

class ABSADataset(Dataset):
    def __init__(self, json_path, tokenizer_name, max_len=128):
        """
        初始化数据集
        :param json_path: JSON 文件路径
        :param tokenizer_name: 预训练模型的名称'roberta-base' 或中文的 'hfl/chinese-roberta-wwm-ext'
        :param max_len: 句子的最大截断长度
        """
        # 加载 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 初始化预训练模型的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_len = max_len

        #初始化句法图建造器
        self.graph_builder = DependencyGraphBuilder()

        # 定义 BIO 标签映射字典
        # O = 0
        # B-NEG = 1, I-NEG = 2
        # B-NEU = 3, I-NEU = 4
        # B-POS = 5, I-POS = 6
        self.polarity_to_tags = {
            0: (1, 2),  # 负向对应的 (B标签, I标签)
            1: (3, 4),  # 中性对应的 (B标签, I标签)
            2: (5, 6)   # 正向对应的 (B标签, I标签)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        aspects = item['aspects']

        # 调用 Tokenizer 进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True, 
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)          # [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)# [max_len]
        offset_mapping = encoding['offset_mapping'].squeeze(0)# [max_len, 2]

        #初始化全为 0 的 Label 张量
        labels = torch.zeros(self.max_len, dtype=torch.long)

        # 将字符级偏移量映射为 Token 标签
        for aspect in aspects:
            char_start = aspect['start_idx']
            char_end = aspect['end_idx']
            pol = aspect['polarity']
            b_tag, i_tag = self.polarity_to_tags[pol]
            
            # 遍历当前句子的所有 Token
            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_start == 0 and offset_end == 0:
                    continue
                
                # 如果当前 Token 的字符区间落在了实体的字符区间内
                if offset_start >= char_start and offset_end <= char_end:
                    # 如果是第一个落在区间内的 Token，打上 B 标签；否则打上 I 标签
                    if offset_start == char_start:
                        labels[i] = b_tag
                    else:
                        labels[i] = i_tag
                        
                # 容错处理：因为子词切分，有时候 offset_start 会比 char_start 稍微大一点
                # （针对英文没空格粘连的情况），这部分可以在后续调优时通过正则进一步优化

        adj_matrix = self.graph_builder.build_adj_matrix(
            text=text, 
            offset_mapping=offset_mapping.tolist(), # 转成 list 传进去
            max_len=self.max_len
        )
        
        # 组装返回字典
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32),
            'text': text  # 返回原文方便后期调试对照
        }
def split_offline_data(input_path, output_dir, seed=42):
    """
    将原始数据切分为 train, val, test 并落盘保存。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 锁死 Python 原生随机种子并打乱数据
    random.seed(seed)
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    # 物理切片
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入硬盘
    splits = {
        "restaurants_train.json": train_data,
        "restaurants_val.json": val_data,
        "restaurants_test.json": test_data
    }
    
    for filename, split_data in splits.items():
        save_path = os.path.join(output_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

# ================= 运行测试模块 =================
if __name__ == "__main__":
    TOKENIZER_NAME = 'roberta-base' 
    
    #json 文件
    current_file_path=os.path.abspath(__file__)
    current_dir_path=os.path.dirname(current_file_path)
    JSON_PATH = os.path.abspath(os.path.join(current_dir_path, "../data/processed/restaurants.json"))
    OUTPUT_DIR = os.path.join(current_dir_path, "..", "data", "processed")
    
    split_offline_data(JSON_PATH, OUTPUT_DIR)