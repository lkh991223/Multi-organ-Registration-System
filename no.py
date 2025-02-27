def max_efficiency(NPU_capacity, GPU_capacity, N, algorithms):
    # The global variable used for storing the highest efficiency.
    global max_eff
    max_eff = 0

    # Helper function to recursively attempt all possible assignment methods.
def helper(n, NPU_rem, GPU_rem, eff):
    global max_eff
    # If all algorithms have been processed, or the computing power of both hardware units has been exhausted, update the highest efficiency and return.
    if n == N or (NPU_rem <= 0 and GPU_rem <= 0):
        max_eff = max(max_eff, eff)
        return
    # Try to run the current algorithm on the NPU.
    if NPU_rem >= algorithms[n][0]:
        helper(n+1, NPU_rem-algorithms[n][0], GPU_rem, eff+algorithms[n][1])
    if GPU_rem >= algorithms[n][0]:ry
        helper(n+1, NPU_rem, GPU_rem-algorithms[n][0], eff+algorithms[n][2])
        helper(n+1, NPU_rem, GPU_rem, eff)

    helper(0, NPU_capacity, GPU_capacity, 0)
    return max_eff

# test sample
NPU_capacity = 30
GPU_capacity = 15
N = 5
algorithms = [[5,3,2],[5,1,3],[14,2,4],[21,5,3],[2,2,3]]

# print output
print(max_efficiency(NPU_capacity, GPU_capacity, N, algorithms))
