import spacy
import numpy as np
import torch

class DependencyGraphBuilder:
    def __init__(self):
        """
        初始化句法依赖图构建器
        """
        print("正在加载 spaCy 英文语言模型 (en_core_web_sm) ...")
        # 加载英语模型。如果以后做中文，可以换成 'zh_core_web_sm'
        self.nlp = spacy.load("en_core_web_sm")

    def build_adj_matrix(self, text, offset_mapping, max_len):
        """
        核心逻辑：将文本的句法树转化为 RoBERTa Token 级别的邻接矩阵
        
        :param text: 原始文本字符串
        :param offset_mapping: RoBERTa 分词器输出的字符级偏移量映射
        :param max_len: 句子的最大截断长度
        :return: 形状为 [max_len, max_len] 的 numpy 矩阵
        """
        # 1. 初始化全为 0 的邻接矩阵
        adj_matrix = np.zeros((max_len, max_len), dtype=np.float32)
        
        # 2. 为每个 Token 添加“自环” (Self-loop)
        # 在图卷积中，一个词在吸收别人特征的同时，必须保留自己的特征，所以对角线全为 1
        for i in range(max_len):
            adj_matrix[i, i] = 1.0

        # 3. 让 spaCy 解析原始文本，生成包含句法依赖关系的 doc 对象
        doc = self.nlp(text)
        
        # 4. 【核心对齐算法】：建立 RoBERTa Token 索引 -> spaCy Token 的映射
        roberta_to_spacy = {}
        
        for roberta_idx, (start, end) in enumerate(offset_mapping):
            # 跳过 PAD, CLS, SEP 等特殊标记 (它们的 start 和 end 都是 0)
            if start == 0 and end == 0:
                continue
                
            # 遍历 spaCy 切出来的真实单词
            for spacy_token in doc:
                spacy_start = spacy_token.idx
                spacy_end = spacy_start + len(spacy_token)
                
                # 如果 RoBERTa 碎片的字符区间，落在了 spaCy 单词的字符区间内
                # 说明这个碎片属于这个单词！
                if start >= spacy_start and end <= spacy_end:
                    roberta_to_spacy[roberta_idx] = spacy_token
                    break # 找到了就跳出内层循环

        # 5. 根据 spaCy 的句法树，在矩阵中连线 (打通 GCN 的通道)
        for i in range(max_len):
            for j in range(max_len):
                # 只有当两个 RoBERTa Token 都找到了对应的真实单词时，才考虑连线
                if i in roberta_to_spacy and j in roberta_to_spacy:
                    token_i = roberta_to_spacy[i]
                    token_j = roberta_to_spacy[j]
                    
                    # 检查句法依赖关系：如果 token_i 是 token_j 的“父亲”，或者反过来
                    if token_i.head == token_j or token_j.head == token_i:
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0 # ABSA 中通常使用无向图，信息双向流通
        # 建立 spacy token -> list of roberta indices
        token_to_roberta = {}
        for roberta_idx, token in roberta_to_spacy.items():
            token_to_roberta.setdefault(token, []).append(roberta_idx)

        # 组内全连接：同一单词的子词之间两两相连
        for indices in token_to_roberta.values():
            for i in indices:
                for j in indices:
                    if i != j:
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0
        
                        
        return adj_matrix

# ================= 运行测试模块 =================
if __name__ == "__main__":
    # 为了测试这个建造器，我们需要手动模拟一下 Tokenizer 的输出
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    
    text = "The screen is great but the battery is bad."
    max_len = 16
    
    encoding = tokenizer(
        text, 
        max_length=max_len, 
        padding='max_length', 
        truncation=True, 
        return_offsets_mapping=True
    )
    
    offset_mapping = encoding['offset_mapping']
    
    # 实例化我们的造桥大师
    builder = DependencyGraphBuilder()
    
    # 提取矩阵
    adj_matrix = builder.build_adj_matrix(text, offset_mapping, max_len)
    
    print("\n====== 句法邻接矩阵 (Adjacency Matrix) 提取结果 ======")
    print(f"原始文本: {text}")
    print(f"切分后的 Tokens: {tokenizer.convert_ids_to_tokens(encoding['input_ids'])}")
    print(f"\n矩阵形状: {adj_matrix.shape}")
    print(adj_matrix)
    
    # 验证一下：屏幕(screen)和惊艳(great)是否连通了？
    # 在这个测试里，screen 是第 2 个 Token，great 是第 4 个 Token
    print(f"\n[screen] 和 [great] 是否在句法上直接相连？: {'是' if adj_matrix[2, 4] == 1 else '否'}")