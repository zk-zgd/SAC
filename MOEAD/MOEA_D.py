#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 11:50
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MOEA_D.py
# @Statement : Multi-objective evolutionary algorithm based on decomposition (MOEA/D)
# @Reference : Zhang Q, Li H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition[J]. IEEE Transactions on Evolutionary Computation, 2007, 11(6): 712-731.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
import pygmo as pg
import time
# DTLZ1
# def cal_obj(x):
#     m = 3  # 目标数量
#     k = len(x) - m + 1  # k 是决策向量 x_M 的维度
#     x_M = x[-k:]  # x_M 是 x 的后 k 个元素
#     g = 100 * (k + np.sum((x_M - 0.5)**2 - np.cos(20.0 * np.pi * (x_M - 0.5))))
#     f = [0.5 * np.prod(x[:m-i]) * (1 + g) if i != (m - 1) else 0.5 * (1 + g) for i in range(m)]
#     return f
#####################################################################################
# DTLZ2
# def g_function(x, m):
#     k = len(x) - m
#     x_m = x[-k:]
#     return np.sum((x_m - 0.5)**2)

# def f1(x, m):
#     g = g_function(x, m)
#     return (1 + g) * np.prod(np.cos(x[:m-1] * np.pi / 2))

# def f2(x, m):
#     g = g_function(x, m)
#     return (1 + g) * np.sin(x[0] * np.pi / 2) * np.prod(np.cos(x[1:m-1] * np.pi / 2))

# def f3(x, m):
#     g = g_function(x, m)
#     return (1 + g) * np.sin(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2)

# def cal_obj(x, m=3):
#     return [f1(x,m), f2(x,m), f3(x,m)] 

###############################################################################
# DTLZ3
def g_function(x, m):
    k = len(x) - m
    x_m = x[-k:]
    return 100 * (k + np.sum((x_m - 0.5)**2 - np.cos(20 * np.pi * (x_m - 0.5))))

def f1(x, m):
    g = g_function(x, m)
    return (1 + g) * np.cos(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2)

def f2(x, m):
    g = g_function(x, m)
    return (1 + g) * np.cos(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2)

def f3(x, m):
    g = g_function(x, m)
    return (1 + g) * np.sin(x[0] * np.pi / 2)

def cal_obj(x, m=3):
    return [f1(x,m), f2(x,m), f3(x,m)] 

###############################################################################
def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def combination(n, m):
    # choose m elements from a n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, dim):
    # 生成大约 npop 个在 dim 维空间上均匀分布的参考点
    h1 = 0
    while combination(h1 + dim, dim - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + dim), dim - 1))) - np.arange(dim - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < dim:
        h2 = 0
        while combination(h1 + dim - 1, dim - 1) + combination(h2 + dim, dim - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + dim), dim - 1))) - np.arange(dim - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * dim)
            points = np.concatenate((points, temp_points), axis=0)
    points = np.where(points != 0, points, 1e-3)
    return points


def crossover(parent1, parent2, lb, ub, dim, pc, eta_c):
    # simulated binary crossover (SBX)
    if np.random.random() < pc:
        beta = np.zeros(dim)
        mu = np.random.random(dim)
        flag1 = mu <= 0.5
        flag2 = ~flag1
        beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
        beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
        offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
        offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
        offspring1 = np.where(((offspring1 >= lb) & (offspring1 <= ub)), offspring1, np.random.uniform(lb, ub))
        offspring2 = np.where(((offspring2 >= lb) & (offspring2 <= ub)), offspring2, np.random.uniform(lb, ub))
        return offspring1 if np.random.random() < 0.5 else offspring2
    else:
        return parent1 if np.random.random() < 0.5 else parent2


def mutation(individual, lb, ub, dim, pm, eta_m):
    # polynomial mutation
    if np.random.random() < pm:
        site = np.random.random(dim) < 1 / dim
        mu = np.random.random(dim)
        delta1 = (individual - lb) / (ub - lb)
        delta2 = (ub - individual) / (ub - lb)
        temp = np.logical_and(site, mu <= 0.5)
        individual[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
        temp = np.logical_and(site, mu > 0.5)
        individual[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
        individual = np.where(((individual >= lb) & (individual <= ub)), individual, np.random.uniform(lb, ub))
    return individual


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, T=20, pc=1, pm=1, eta_c=20, eta_m=20):
    start_time = time.time()
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space:7
    nobj = len(cal_obj((lb + ub) / 2))  # the dimension of objective space
    V = reference_points(npop, nobj)  # 用于将MOA分解为多个SOA。权重向量V定义了每个子问题的优化方向
    sigma = squareform(pdist(V, metric='euclidean'), force='no', checks=True)  # 使用欧氏距离计算每个权重向量间的距离
    B = np.argsort(sigma)[:, : T]  # 对于每个子问题，找到其权重向量最近的T个其他子问题的索引
    # npop = V.shape[0]  # population size
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = np.array([cal_obj(x) for x in pop])  # objectives
    z = np.min(objs, axis=0)  # ideal point
    hv_values = []
    ref_point = np.max(objs, axis=0) + 0.1  # Initial reference point slightly higher than the max observed

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 200 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        for i in range(npop):

            # Step 2.1. Crossover + mutation
            [p1, p2] = np.random.choice(B[i], 2, replace=False)
            off = crossover(pop[p1], pop[p2], lb, ub, nvar, pc, eta_c)
            off = mutation(off, lb, ub, nvar, pm, eta_m)
            off_obj = cal_obj(off)

            # Step 2.2. Update the ideal point：记录每个目标的最小值，用于帮助计算解与理想点的距离，影响适应度计算
            z = np.min((z, off_obj), axis=0)

            # Step 2.3. Update neighbor solutions：对于每个子问题的每个邻居，如果新生成的解相对于该邻居的当前解更优（使用加权Tchebycheff方法评估），则替换该邻居的解。
            for j in B[i]:
                if np.max(V[j] * np.abs(off_obj - z)) < np.max(V[j] * np.abs(objs[j] - z)):
                    pop[j] = off
                    objs[j] = off_obj
        
        # Update the reference point dynamically
        max_obj = np.max(objs, axis=0)
        ref_point = np.maximum(ref_point, max_obj + 0.1)  # Ensure reference point is always above the observed max

        # Calculating hypervolume
        non_dominated_mask = np.array([not any(np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]) for j in range(npop) if i != j) for i in range(npop)])
        current_pf = objs[non_dominated_mask]
        hv = pg.hypervolume(current_pf)
        current_hv = hv.compute(ref_point)
        hv_values.append(current_hv)
    elapsed_time = time.time() - start_time  # End timing
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    max_hv = max(hv_values)
    min_hv = min(hv_values)
    average_hv = sum(hv_values) / len(hv_values)

    print(f"最高HV值: {max_hv}")
    print(f"最低HV值: {min_hv}")
    print(f"平均HV值: {average_hv}")
    # Step 3. Sort the results：确定最终种群中的非支配解
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    
    # Plot hypervolume over generations
    plt.figure()
    plt.plot(hv_values, label='Hypervolume')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = [o[0] for o in pf]
    # y = [o[1] for o in pf]
    # z = [o[2] for o in pf]
    # ax.scatter(x,y,z)
    # ax.set_xlabel('Objective 1')
    # ax.set_ylabel('Objective 2')
    # ax.set_zlabel('Objective 3')
    # plt.title('The Pareto front of ZDT3')
    # plt.savefig('Pareto front')
    # plt.show()


if __name__ == '__main__':
    main(21, 921, np.array([0] * 7), np.array([1] * 7))
