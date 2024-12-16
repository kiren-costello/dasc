import torch
import numpy as np


def pad_and_tensorize_adjacency_matrices(adj_matrices):
    if not adj_matrices:
        return torch.tensor([]), torch.tensor([])
    max_size = max(len(mat) for mat in adj_matrices)

    # 填充每个矩阵到最大尺寸
    padded_matrices = []
    masks = []

    for mat in adj_matrices:
        num_nodes = len(mat)
        pad_size = max_size - num_nodes
        padded = np.pad(mat, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
        padded_matrices.append(padded)

        mask = torch.zeros(max_size)
        mask[:num_nodes] = 1
        masks.append(mask)

    padded_matrices_array = np.array(padded_matrices)

    # 转换为张量
    tensor_adj = torch.from_numpy(padded_matrices_array)

    tensor_masks = torch.stack(masks)

    return tensor_adj, tensor_masks


# def pad_and_tensorize_embeddings(embeddings_list):
#
#     max_nodes = max(len(graph_emb) for graph_emb in embeddings_list)
#
#
#     emb_dim = len(embeddings_list[0][0])
#
#
#     padded_embeddings = []
#     for graph_emb in embeddings_list:
#         pad_size = max_nodes - len(graph_emb)
#         padded = np.pad(graph_emb, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
#         padded_embeddings.append(padded)
#
#
#     padded_embeddings_array = np.array(padded_embeddings)
#
#
#     tensor_emb = torch.from_numpy(padded_embeddings_array).float()
#
#     return tensor_emb

def pad_and_tensorize_embeddings(embeddings_list):
    if not embeddings_list:
        return torch.tensor([])

    if not has_number(embeddings_list):
        return torch.tensor([])
    # 找到最大的节点数量
    max_nodes = max(len(graph_emb) for graph_emb in embeddings_list if graph_emb)

    # 找到嵌入维度，假设嵌入列表中至少有一个非空嵌入
    # emb_dim = len(embeddings_list[0][0]) if embeddings_list[0] else len(embeddings_list[1][0])
    for item in embeddings_list:
        if has_number(item):
            emb_dim = len(item[0])
            break

    padded_embeddings = []
    for graph_emb in embeddings_list:
        if graph_emb:
            pad_size = max_nodes - len(graph_emb)
            padded = np.pad(graph_emb, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        else:
            padded = np.zeros((max_nodes, emb_dim))  # 处理空列表的情况
        padded_embeddings.append(padded)

    padded_embeddings_array = np.array(padded_embeddings)

    tensor_emb = torch.from_numpy(padded_embeddings_array).float()

    return tensor_emb

def has_number(lst):
    for item in lst:
        if isinstance(item, list):
            if has_number(item):  # 递归检查子列表
                return True
        else:
            if isinstance(item, (int, float)):  # 判断是否为数字类型
                return True
    return False
if __name__ == "__main__":
    # 示例邻接矩阵列表
    adj_matrices = [
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    ]

    # 调用函数
    tensor_adj, tensor_masks = pad_and_tensorize_adjacency_matrices(adj_matrices)

    print("Adjacency Tensor:")
    print(tensor_adj)
    print("Adjacency Tensor Shape:", tensor_adj.shape)

    print("\nMask Tensor:")
    print(tensor_masks)
    print("Mask Tensor Shape:", tensor_masks.shape)

# if __name__ == "__main__":
#     # 示例图嵌入列表
#     embeddings_list = [
#         [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
#         [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]],
#         [[1.6, 1.7, 1.8], [1.9, 2.0, 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]]
#     ]
#
#     # 调用函数
#     result_tensor = pad_and_tensorize_embeddings(embeddings_list)
#
#     print(result_tensor)
#     print(result_tensor.shape)
