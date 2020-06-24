import numpy as np
from collections import defaultdict

# 1. softmax
def softmax(lst):
    exp_sum = np.sum(np.exp(lst))
    return np.exp(lst)/exp_sum

def sum_lst(lst, target):
    map = {}
    for i, n in enumerate(lst):
        try:
            map[target - n]
            return n, target - n
        except:
            map[n] = True
    return -1

def dup_zero(lst):
    len_lst = len(lst)
    flag = False
    for i, n in enumerate(lst):
        if flag:
            flag = False
            continue
        if n == 0:
            for j in range(len_lst-1, i + 1, -1):
                print(j)
                lst[j] = lst[j - 1]
            try:
                lst[i + 1] = lst[i]
            except:
                pass

            flag = True

# num_ways

table = {}
def jump(num_moves, target):
    try:
        return table[target]
    except:
        # print(target)
        if target == 0:
            return 1
        if target < 0:
            return 0
        num_ways = 0
        for i in num_moves:
            num_ways += jump(num_moves, target - i)
        table[target] = num_ways
    return num_ways



if __name__ == '__main__':
    # print(softmax([3.0, 1.0, 0.2]))
    # print(sum_lst([1,3,5,6,19,10], 13))
    print(jump([1, 3, 5], 6))