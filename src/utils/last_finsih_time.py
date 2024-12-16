from collections import deque


def calculate_latest_finish_time(A, D, T):
    n = len(A)
    in_degree = [0] * n
    LFT = [T] * n

    # Step 1: Compute in-degrees
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:
                in_degree[j] += 1

    # Step 2: Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topological_order = []

    while queue:
        node = queue.popleft()
        topological_order.append(node)
        for j in range(n):
            if A[node][j] == 1:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    # Step 3: Calculate Latest Finish Time (LFT) in reverse topological order
    for node in reversed(topological_order):
        for j in range(n):
            if A[node][j] == 1:
                LFT[node] = min(LFT[node], LFT[j] - D[j])

    return LFT


if __name__ =="__main__":
    A = [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    D = [3, 2, 4, 1, 2]
    T = 10

    # Calculate Latest Finish Time
    LFT = calculate_latest_finish_time(A, D, T)

    # Output the result
    print("Latest Finish Times for each task:", LFT)