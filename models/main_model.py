import torch
import torch.nn as nn
from transformers import AutoModel

# 导入我们之前一步步写好并测试通过的零件
from models.layers.gcn_layer import GCNLayer
from models.layers.attention import SelfAttentionLayer
from models.layers.crf_layer import CRFLayer

class ABSAMainModel(nn.Module):
    # 🌟 新增：参数列表里加入 use_gcn 和 use_attn
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
        
        # 实例化图卷积层
        self.gcn = GCNLayer(
            in_features=self.hidden_size, 
            out_features=gcn_out_dim, 
            dropout_rate=dropout_rate
        )
        
        # 🌟 核心修改：动态适应分类器的输入维度
        # 如果用 GCN，维度就是 gcn_out_dim (300)；如果不用 GCN，维度就是 roberta 出来的 hidden_size (768)
        classifier_in_dim = gcn_out_dim if self.use_gcn else self.hidden_size
        self.classifier = nn.Linear(classifier_in_dim, num_tags)
        
        self.crf = CRFLayer(num_tags=num_tags)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, adj_matrix, labels=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 🌟 根据开关决定是否走 Attention
        if self.use_attn:
            attn_output = self.attention(sequence_output, attention_mask=attention_mask)
        else:
            attn_output = sequence_output # 关掉就直接短路跳过
            
        # 🌟 根据开关决定是否走 GCN
        if self.use_gcn:
            final_features = self.gcn(attn_output, adj_matrix)
        else:
            final_features = attn_output # 关掉就直接短路跳过
            
        # 映射到标签空间
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
    TOKENIZER_NAME = 'roberta-base' # 测试用英文，若网络不通可换成本地路径
    
    # 1. 实例化主模型
    model = ABSAMainModel(model_name_or_path=TOKENIZER_NAME, num_tags=NUM_TAGS, gcn_out_dim=300)
    
    # 2. 伪造输入张量 (完全模拟我们从 Dataset.__getitem__ 拼出的 Batch 数据)
    dummy_input_ids = torch.randint(10, 5000, (BATCH_SIZE, SEQ_LEN))
    dummy_attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # 第一句话长 10，后 2 个是 PAD
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 第二句话长 8，后 4 个是 PAD
    ], dtype=torch.long)
    
    dummy_adj_matrix = torch.randn(BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    dummy_labels = torch.randint(0, NUM_TAGS, (BATCH_SIZE, SEQ_LEN))
    
    print("\n====== 模拟训练模式 (Forward with Labels) ======")
    model.train() # 切换为训练模式，激活 Dropout
    loss = model(
        input_ids=dummy_input_ids, 
        attention_mask=dummy_attention_mask, 
        adj_matrix=dummy_adj_matrix, 
        labels=dummy_labels
    )
    print(f"训练阶段前向传播成功！算出的批次平均 CRF Loss: {loss.item():.4f}")
    
    print("\n====== 模拟预测模式 (Forward without Labels) ======")
    model.eval() # 切换为评估模式，关闭 Dropout
    with torch.no_grad(): # 关闭梯度上下文，节省显存
        preds = model(
            input_ids=dummy_input_ids, 
            attention_mask=dummy_attention_mask, 
            adj_matrix=dummy_adj_matrix, 
            labels=None # 不传标签
        )
    print(f"预测阶段前向传播成功！")
    print(f"第一句话的预测路径长度 (应为 10): {len(preds[0])} -> {preds[0]}")
    print(f"第二句话的预测路径长度 (应为 8): {len(preds[1])} -> {preds[1]}")
    
    print("\n🎉 宏伟蓝图组装完毕！从数据输入到 Loss 产出/解码输出的端到端大管道已完全打通！")