import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# 导入我们之前一步步写好并测试通过的零件
from models.layers.gcn_layer import GCNLayer
from models.layers.attention import SelfAttentionLayer
from models.layers.crf_layer import CRFLayer

# =====================================================================
# 🌟 新增：动态图学习器 (Latent Dynamic Graph Learner)
# =====================================================================
class DynamicGraphLearner(nn.Module):
    def __init__(self, hidden_dim, initial_alpha=0.5):
        super(DynamicGraphLearner, self).__init__()
        # 用于计算纯语义相似度矩阵的 Q 和 K
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        # 核心：可学习的融合门控标量，初始倾向设为 initial_alpha
        self.gate = nn.Parameter(torch.tensor([initial_alpha]))

    def forward(self, hidden_states, static_adj, attention_mask):
        # 1. 计算当前语境下的语义关联图
        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        semantic_scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_states.size(-1) ** 0.5)
        
        # 屏蔽 PAD 标记
        extended_mask = attention_mask.unsqueeze(1)
        semantic_scores = semantic_scores.masked_fill(extended_mask == 0, -1e9)
        semantic_adj = F.softmax(semantic_scores, dim=-1)
        
        # 2. 动态融合 (Sigmoid 确保权重在 0~1 之间)
        g = torch.sigmoid(self.gate)
        dynamic_adj = g * static_adj + (1 - g) * semantic_adj
        
        return dynamic_adj

# =====================================================================
# 主模型
# =====================================================================
class ABSAMainModel(nn.Module):
    def __init__(self, model_name_or_path, num_tags=7, gcn_out_dim=300, dropout_rate=0.1, use_gcn=True, use_attn=True):
        super(ABSAMainModel, self).__init__()
        
        self.use_gcn = use_gcn
        self.use_attn = use_attn
        
        print(f"正在加载预训练语言模型权重: {model_name_or_path} ...")
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.roberta.config.hidden_size
        
        # 实例化自注意力层
        self.attention = SelfAttentionLayer(
            hidden_size=self.hidden_size, 
            num_heads=8, 
            dropout_rate=dropout_rate
        )
        
        # 🌟 新增：实例化动态图学习器 (输入维度固定为 RoBERTa 的 hidden_size)
        self.graph_learner = DynamicGraphLearner(hidden_dim=self.hidden_size)
        
        # 实例化图卷积层
        self.gcn = GCNLayer(
            in_features=self.hidden_size, 
            out_features=gcn_out_dim, 
            dropout_rate=dropout_rate
        )
        
        # 动态适应分类器的输入维度
        classifier_in_dim = gcn_out_dim if self.use_gcn else self.hidden_size
        self.classifier = nn.Linear(classifier_in_dim, num_tags)
        
        self.crf = CRFLayer(num_tags=num_tags)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, adj_matrix, labels=None):
        # 1. 底座特征提取
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 2. 多头自注意力模块
        if self.use_attn:
            attn_output = self.attention(sequence_output, attention_mask=attention_mask)
        else:
            attn_output = sequence_output # 短路跳过
            
        # 3. 图卷积网络模块 (含动态图学习)
        if self.use_gcn:
            # 🌟 核心修改：在送入 GCN 之前，拦截静态 adj_matrix，将其升级为动态矩阵！
            # 注意参数传递：隐状态使用 attn_output（包含全局信息的最新特征）
            dynamic_adj = self.graph_learner(attn_output, static_adj=adj_matrix, attention_mask=attention_mask)
            
            # GCN 吃进去的是经过融合的 dynamic_adj，不再是死板的句法树了
            final_features = self.gcn(attn_output, dynamic_adj)
        else:
            final_features = attn_output # 短路跳过
            
        # 4. 发射矩阵与 CRF 解码
        emissions = self.classifier(final_features)
        
        if labels is not None:
            loss = self.crf(emissions, tags=labels, mask=attention_mask)
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask)
            return predictions

# ================= 司令部整装运行测试 =================
if __name__ == "__main__":
    # 模拟超参数
    BATCH_SIZE = 2
    SEQ_LEN = 12
    NUM_TAGS = 7
    TOKENIZER_NAME = 'roberta-base'
    
    # 1. 实例化主模型
    model = ABSAMainModel(model_name_or_path=TOKENIZER_NAME, num_tags=NUM_TAGS, gcn_out_dim=300)
    
    # 2. 伪造输入张量
    dummy_input_ids = torch.randint(10, 5000, (BATCH_SIZE, SEQ_LEN))
    dummy_attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  
    ], dtype=torch.long)
    
    # 静态先验句法树 (spaCy 吐出来的数据)
    dummy_adj_matrix = torch.randn(BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    dummy_labels = torch.randint(0, NUM_TAGS, (BATCH_SIZE, SEQ_LEN))
    
    print("\n====== 模拟训练模式 (Forward with Labels) ======")
    model.train() 
    loss = model(
        input_ids=dummy_input_ids, 
        attention_mask=dummy_attention_mask, 
        adj_matrix=dummy_adj_matrix, 
        labels=dummy_labels
    )
    print(f"训练阶段前向传播成功！算出的批次平均 CRF Loss: {loss.item():.4f}")
    
    print("\n====== 模拟预测模式 (Forward without Labels) ======")
    model.eval() 
    with torch.no_grad():
        preds = model(
            input_ids=dummy_input_ids, 
            attention_mask=dummy_attention_mask, 
            adj_matrix=dummy_adj_matrix, 
            labels=None 
        )
    print(f"预测阶段前向传播成功！")
    print(f"第一句话的预测路径长度 (应为 10): {len(preds[0])} -> {preds[0]}")
    print(f"第二句话的预测路径长度 (应为 8): {len(preds[1])} -> {preds[1]}")
    print("\n🎉 带有动态图结构的端到端系统组装完毕！")