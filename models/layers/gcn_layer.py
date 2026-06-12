import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        """
        初始化单层图卷积网络 (Graph Convolutional Network Layer)
        
        数学公式: H^(l+1) = ReLU(A * H^(l) * W + b)
        
        :param in_features: 输入特征的维度大小 (例如 RoBERTa-base 输出维度通常是 768)
        :param out_features: 经过 GCN 映射后的输出特征维度大小
        :param dropout_rate: 防止模型过拟合的丢弃率
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. 定义可学习的权重矩阵 W 和偏置向量 b
        # 使用 nn.Parameter 包装，将其注册为模型的参数，以便在反向传播时自动计算梯度
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # 2. 定义 Dropout 层，随机丢弃部分神经元，增强模型鲁棒性
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化策略：
        使用 Xavier 均匀分布初始化权重 W，将偏置 b 初始化为 0。
        这能有效避免深度网络在训练初期的梯度消失或梯度爆炸问题。
        """
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, text_features, adj_matrix):
        """
        前向传播逻辑
        
        :param text_features: 文本特征矩阵 H。
                              形状: [batch_size, seq_len, in_features]
        :param adj_matrix: 句法依赖邻接矩阵 A。
                           形状: [batch_size, seq_len, seq_len]
        :return: 图聚合后的新特征矩阵。
                 形状: [batch_size, seq_len, out_features]
        """
        # 1. 确保自环存在（如果已存在，I 叠加会变成 2，但无大碍，后续归一化会处理）
        I = torch.eye(adj_matrix.size(1), device=adj_matrix.device).unsqueeze(0)
        adj = adj_matrix + I   # 此时对角线为 2（如果原为1）或 1（原为0）
        
        # 2. 对称归一化：D^{-0.5} * A * D^{-0.5}
        rowsum = adj.sum(dim=-1)   # [batch, seq_len]
        d_inv_sqrt = torch.pow(rowsum + 1e-9, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # 3. 图卷积
        support = torch.matmul(text_features, self.weight)
        output = torch.matmul(norm_adj, support)
        output = output + self.bias
        output = torch.relu(output)
        return self.dropout(output)

# ================= 测试模块 =================
if __name__ == "__main__":
    # 模拟超参数
    batch_size = 4
    seq_len = 32
    in_features = 768  # RoBERTa 的默认特征维度
    out_features = 300 # 希望 GCN 提取出的核心特征维度
    
    # 实例化 GCN 层
    gcn_layer = GCNLayer(in_features=in_features, out_features=out_features)
    
    # 伪造输入数据 (随机生成)
    # 1. 模拟 RoBERTa 输出的动态词向量
    dummy_text_features = torch.randn(batch_size, seq_len, in_features)
    
    # 2. 模拟从 spaCy 提取出的句法邻接矩阵
    # 通常邻接矩阵是一个稀疏的 0/1 矩阵，这里用随机数模拟
    dummy_adj_matrix = torch.randn(batch_size, seq_len, seq_len)
    
    # 前向传播
    output_features = gcn_layer(dummy_text_features, dummy_adj_matrix)
    
    print("\n====== GCN 层输出维度 ======")
    print(f"聚合后特征维度: {output_features.shape}")
    
    assert output_features.shape == (batch_size, seq_len, out_features), "输出维度与预期不符！"
    print("\n🎉 测试通过！GCN 层维度推导完全正确！")

# # ===================================================   GAT  ====================================================================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GCNLayer(nn.Module):
#     """
#     图注意力层 (GAT) 实现，但保留类名 GCNLayer 以兼容现有代码
#     数学逻辑: H' = Attention(Q, K, V) 基于邻接矩阵屏蔽
#     """
#     def __init__(self, in_features, out_features, dropout_rate=0.1):
#         super(GCNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # 线性变换 W
#         self.W = nn.Linear(in_features, out_features, bias=False)
#         # 注意力参数 a (拼接特征后映射到标量)
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.dropout = nn.Dropout(dropout_rate)
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W.weight)
#         nn.init.xavier_uniform_(self.a)

#     def forward(self, text_features, adj_matrix):
#         """
#         text_features: [batch_size, seq_len, in_features]
#         adj_matrix: [batch_size, seq_len, seq_len]   (0/1 矩阵，含自环)
#         返回: [batch_size, seq_len, out_features]
#         """
#         batch_size, seq_len, _ = text_features.shape
        
#         # 1. 线性变换
#         h = self.W(text_features)  # [batch, seq_len, out_features]
        
#         # 2. 计算注意力系数 e_ij
#         # 扩展 h 以便计算所有 (i,j) 对
#         h_i = h.unsqueeze(2)  # [batch, seq_len, 1, out_features]
#         h_j = h.unsqueeze(1)  # [batch, 1, seq_len, out_features]
#         # 拼接 [h_i || h_j]
#         h_concat = torch.cat([h_i.expand(-1, -1, seq_len, -1),
#                               h_j.expand(-1, seq_len, -1, -1)], dim=-1)  # [b, L, L, 2*out]
#         # 计算 e_ij = LeakyReLU(a^T [h_i || h_j])
#         e = torch.matmul(h_concat, self.a).squeeze(-1)  # [batch, seq_len, seq_len]
#         e = self.leaky_relu(e)
        
#         # 3. 根据邻接矩阵屏蔽 (adj_matrix 中为 0 的位置设为负无穷)
#         # 注意：adj_matrix 已经是 0/1 矩阵，对角线为 1（自环）
#         masked_e = e.masked_fill(adj_matrix == 0, -1e9)
        
#         # 4. Softmax 归一化得到注意力权重
#         alpha = F.softmax(masked_e, dim=-1)  # [batch, seq_len, seq_len]
#         alpha = self.dropout(alpha)
        
#         # 5. 加权聚合
#         h_prime = torch.matmul(alpha, h)  # [batch, seq_len, out_features]
#         h_prime = self.dropout(h_prime)
        
#         return h_prime


# # ================================================================================bilstm====================================
# import torch
# import torch.nn as nn

# class GCNLayer(nn.Module):
#     """
#     标准图卷积层 (GCN)
#     包含：自环添加、对称归一化、线性变换、ReLU、Dropout
#     输入形状: [batch_size, seq_len, in_features]
#     邻接矩阵: [batch_size, seq_len, seq_len]  (0/1 值，可包含自环)
#     输出形状: [batch_size, seq_len, out_features]
#     """
#     def __init__(self, in_features, out_features, dropout_rate=0.1):
#         super(GCNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # 可学习的权重矩阵和偏置
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         self.dropout = nn.Dropout(dropout_rate)
        
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         # Xavier 初始化
#         nn.init.xavier_uniform_(self.weight)
#         nn.init.zeros_(self.bias)
    
#     def forward(self, text_features, adj_matrix):
#         """
#         text_features: [batch_size, seq_len, in_features]
#         adj_matrix: [batch_size, seq_len, seq_len] (原始邻接矩阵，推荐已包含自环，但本层会再次添加)
#         """
#         batch_size, seq_len, _ = text_features.shape
#         device = text_features.device
        
#         # 1. 添加自环 (确保每个节点保留自身信息)
#         I = torch.eye(seq_len, device=device).unsqueeze(0)  # [1, L, L]
#         adj = adj_matrix + I   # 若原矩阵已有自环，对角线变为2，但后续归一化会调整
        
#         # 2. 对称归一化: D^{-0.5} * A * D^{-0.5}
#         rowsum = adj.sum(dim=-1)          # [batch, L]
#         d_inv_sqrt = torch.pow(rowsum + 1e-9, -0.5)   # [batch, L]
#         d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt) # [batch, L, L]
#         norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt  # [batch, L, L]
        
#         # 3. 线性变换
#         support = torch.matmul(text_features, self.weight)  # [batch, L, out_features]
        
#         # 4. 图卷积: A' * H * W
#         output = torch.matmul(norm_adj, support)
#         output = output + self.bias
#         output = torch.relu(output)
#         output = self.dropout(output)
        
#         return output