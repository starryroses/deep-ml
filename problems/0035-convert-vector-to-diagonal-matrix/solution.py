import numpy as np

def make_diagonal(x):
    n = len(x)
    ans = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        ans[i][i] = x[i]
    return ans