def max_efficiency(NPU_capacity, GPU_capacity, N, algorithms):
    # 用于存储最高效率的全局变量
    global max_eff
    max_eff = 0

    # 辅助函数，用于递归地尝试所有分配方式
def helper(n, NPU_rem, GPU_rem, eff):
    global max_eff
    # 如果已经处理完所有算法，或者两个硬件的算力都用尽了，更新最高效率并返回
    if n == N or (NPU_rem <= 0 and GPU_rem <= 0):
        max_eff = max(max_eff, eff)
        return
    # 尝试在NPU上运行当前算法
    if NPU_rem >= algorithms[n][0]:
        helper(n+1, NPU_rem-algorithms[n][0], GPU_rem, eff+algorithms[n][1])
    if GPU_rem >= algorithms[n][0]:
        helper(n+1, NPU_rem, GPU_rem-algorithms[n][0], eff+algorithms[n][2])
        helper(n+1, NPU_rem, GPU_rem, eff)

    helper(0, NPU_capacity, GPU_capacity, 0)
    return max_eff

# 测试用例
NPU_capacity = 30
GPU_capacity = 15
N = 5
algorithms = [[5,3,2],[5,1,3],[14,2,4],[21,5,3],[2,2,3]]

# 调用函数并打印结果
print(max_efficiency(NPU_capacity, GPU_capacity, N, algorithms))