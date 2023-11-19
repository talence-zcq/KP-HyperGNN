import numpy as np
from scipy.sparse import coo_matrix

# 假设 A 和 B 是两个相同形状的 Scipy 稀疏矩阵
# 这里假设 A 和 B 是 COO 格式的稀疏矩阵
data_A = np.array([1, 2, 0, 3, 1, 4])
row_A = np.array([0, 1, 1, 2, 2, 3])
col_A = np.array([0, 1, 2, 0, 2, 3])
A = coo_matrix((data_A, (row_A, col_A)), shape=(4, 4))

data_B = np.array([5, 6, 7,7, 8, 9, 10])

row_B = np.array([0, 1, 1,2, 2, 2, 3])
col_B = np.array([0, 1, 2,2, 0, 1, 3])
B = coo_matrix((data_B, (row_B, col_B)), shape=(4, 4))

# 获取 A 中大于 0 的元素的坐标
nonzero_indices = A.nonzero()

# 将 B 中与 A 非零元素相对应的位置置零
for row, col in zip(nonzero_indices[0], nonzero_indices[1]):
    B.data[np.where((B.row == row) & (B.col == col))] = 0

print("Modified B:")
print(B.toarray())
