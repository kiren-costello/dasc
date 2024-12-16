import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, config_of_policy):
        super(PolicyNet, self).__init__()
        self.GAT = GAT(config_of_policy["GAT_input_dim"], config_of_policy["GAT_hidden_dim"],
                       config_of_policy["GAT_output_dim"], config_of_policy["GAT_dropout"],
                       config_of_policy["GAT_alpha"], config_of_policy["GAT_head"])

        self.encoder = MLP(config_of_policy["encoder_input_dim"], config_of_policy["mlp_hidden_dim1"],
                           config_of_policy["mlp_hidden_dim2"], config_of_policy["mlp_output_dim"])

        self.max_nodes = config_of_policy["max_nodes"]

    def forward(self, graph_inf, graph_matrix,masks, node_order, work, subtask):
        # features, adjs, masks = pad_graphs(graph_inf, graph_matrix, self.max_nodes, device)
        # x = self.GAT(features, adjs, masks)
        x = self.GAT(graph_inf, graph_matrix, masks)
        selected_subtask = torch.stack([x[i, node_order[i]] for i in range(x.size(0))])
        inf = torch.cat((selected_subtask, work, subtask), dim=1)
        out = self.encoder(inf)
        out = F.softmax(out, dim=-1)
        return out


class ValueNet(nn.Module):
    def __init__(self, config_of_value):
        super(ValueNet, self).__init__()
        self.worker_encoder = GRUNetwork(config_of_value["worker_embedding_dim"], config_of_value["worker_hidden_dim"],
                                         config_of_value["worker_output_dim"])
        self.subtask_encoder = GRUNetwork(config_of_value["subtask_embedding_dim"],
                                          config_of_value["subtask_hidden_dim"], config_of_value["subtask_output_dim"])
        self.encoder = MLP(config_of_value["encoder_input_dim"], config_of_value["mlp_hidden_dim1"],
                           config_of_value["mlp_hidden_dim2"], config_of_value["mlp_output_dim"])

    def forward(self, worker_embedding, subtask_embedding):
        x = self.worker_encoder(worker_embedding)
        y = self.subtask_encoder(subtask_embedding)
        x = torch.cat((x, y), dim=1)
        x = self.encoder(x)
        return x


# def pad_graphs(graph_inf, graph_matrix, max_nodes, device):
#     padded_features = []
#     padded_adjs = []
#     masks = []
#
#     for features, adj in zip(graph_inf, graph_matrix):
#         num_nodes = features.shape[0]
#
#         # Padding features
#         pad_size = max_nodes - num_nodes
#         padded_features.append(torch.cat([features, torch.zeros(pad_size, features.shape[1]).to(device)], dim=0))
#
#         # Padding adjacency matrix
#         padded_adj = torch.zeros(max_nodes, max_nodes)
#         padded_adj[:num_nodes, :num_nodes] = adj
#         padded_adjs.append(padded_adj)
#
#         # Creating mask
#         mask = torch.zeros(max_nodes)
#         mask[:num_nodes] = 1
#         masks.append(mask)
#
#     return torch.stack(padded_features).to(device), torch.stack(padded_adjs).to(device), torch.stack(masks).to(device)


# 定义GraphAttentionLayer类
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, mask):
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (batch_size, num_nodes, num_nodes, 2 * out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (batch_size, num_nodes, num_nodes)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # (batch_size, num_nodes, num_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (batch_size, num_nodes, out_features)

        # Apply mask
        h_prime = h_prime * mask.unsqueeze(2)  # (batch_size, num_nodes, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, num_nodes, _ = Wh.size()
        Wh_repeated = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # (batch_size, num_nodes, num_nodes, out_features)
        Wh_repeated_transposed = Wh.unsqueeze(1).repeat(1, num_nodes, 1,
                                                        1)  # (batch_size, num_nodes, num_nodes, out_features)
        return torch.cat([Wh_repeated, Wh_repeated_transposed],
                         dim=3)  # (batch_size, num_nodes, num_nodes, 2 * out_features)


# 定义GAT类
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        )

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, mask):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, mask) for att in self.attentions], dim=2)  # (batch_size, num_nodes, nhid * nheads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj, mask)  # (batch_size, num_nodes, nclass)

        # Apply mask
        x = x * mask.unsqueeze(2)  # (batch_size, num_nodes, nclass)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNetwork, self).__init__()
        # 初始化 GRU 层
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # 初始化全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, seq_len, input_dim)
        # 通过 GRU 层处理输入
        _, hidden = self.gru(x)  # hidden 形状为 (num_layers * num_directions, batch_size, hidden_dim)

        # 取最后一层的隐藏状态作为输出（假设使用单层 GRU）
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # 通过全连接层映射到固定维度
        output = self.fc(last_hidden)  # (batch_size, output_dim)
        return output

# class SelfAttention(torch.nn.Module):
#     def __init__(self, embed_size, heads):
#         super(SelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads
#
#         assert (
#                 self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"
#
#         self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
#         self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
#         self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
#         self.fc_out = torch.nn.Linear(embed_size, embed_size)
#
#     def forward(self, value, key, query, mask=None):
#         N = query.shape[0]
#         value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
#
#         # Split embedding into multiple heads
#         values = self.values(value).view(N, value_len, self.heads, self.head_dim)
#         keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
#         queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)
#
#         # Transpose to get dimensions [N, heads, value_len, head_dim]
#         values = values.permute(0, 2, 1, 3)
#         keys = keys.permute(0, 2, 1, 3)
#         queries = queries.permute(0, 2, 1, 3)
#
#         # Scaled dot-product attention
#         energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # [N, heads, query_len, key_len]
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))
#
#         attention = F.softmax(energy / (self.head_dim ** 0.5), dim=3)  # [N, heads, query_len, key_len]
#         out = torch.matmul(attention, values)  # [N, heads, query_len, head_dim]
#
#         out = out.permute(0, 2, 1, 3).contiguous().view(N, query_len, self.embed_size)
#         out = self.fc_out(out)
#
#         return out
