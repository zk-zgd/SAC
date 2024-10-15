# Required Libraries
import copy
import numpy as np
import pygmo as pg
import random
import os
import matplotlib.pyplot as plt
import time
############################################################################
# DTLZ1
# 辅助函数
# def g_function(x, m=12):
#     x = np.array(x)
#     sum_x = np.sum((x[2:m] - 0.5)**2 - np.cos(20.0 * np.pi * (x[2:m] - 0.5)))
#     return 100 * (m - 2 + sum_x)

# def f1(x):
#     return 0.5 * (1 - x[0]) * (1 + g_function(x))

# def f2(x):
#     return 0.5 * (1 - x[1]) * (1 + g_function(x))

# def f3(x):
#     return 0.5 * (1 - x[2]) * (1 + g_function(x))
############################################################################
# DTLZ2
# def dtlz2(x, M):
#     n = len(x)
#     pi = np.pi
#     k = n - M + 1
#     g = sum((xi-0.5)**2 for xi in x[-k:])
#     f = [(1 + g) * np.prod([np.cos(xi * pi / 2) for xi in x[:M-i-1]]) * (np.sin(x[M-i-1] * pi / 2) if i != 0 else 1) for i in range(M)]
#     return f

# def f1(x):
#     return dtlz2(x, 3)[0]

# def f2(x):
#     return dtlz2(x, 3)[1]

# def f3(x):
#     return dtlz2(x, 3)[2]

############################################################################
# DTLZ3
def g_function(x, M):
    k = len(x) - M
    # 使用 np.fromiter 来创建数组
    sum_x = np.sum(np.fromiter(((x[i] - 0.5)**2 - np.cos(20.0 * np.pi * (x[i] - 0.5)) for i in range(M, len(x))), dtype=float))
    return 100 * (k + sum_x)

def dtlz3(x, M):
    pi = np.pi
    g = g_function(x, M)
    f = [(1 + g) * np.prod([np.cos(xi * pi / 2) for xi in x[:M-i-1]]) * (np.sin(x[M-i-1] * pi / 2) if i != 0 else 1) for i in range(M)]
    return f

def f1(x):
    return dtlz3(x, 3)[0]

def f2(x):
    return dtlz3(x, 3)[1]

def f3(x):
    return dtlz3(x, 3)[2]

############################################################################

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [f1, f2, f3]):
    # 创建一个形状为（种群个体数量，个体的决策变量数+目标函数数）
    population = np.zeros((population_size, len(min_values) + len(list_of_functions)))
    # 对每一个种群个体遍历
    for i in range(0, population_size):
        # 对每一个决策变量
        for j in range(0, len(min_values)):
            # 通过决策变量范围确定决策变量的位置：一个指定范围内的随机值
            population[i,j] = random.uniform(min_values[j], max_values[j]) 
        # 对每一个目标函数     
        for k in range (1, len(list_of_functions) + 1):
            # 计算目标函数值，并存储在个体i的最后几列；-k是从后到前的索引
            # population[i,-k] = list_of_functions[-k](list(population[i,0:population.shape[1]-len(list_of_functions)]))
            population[i, -k] = list_of_functions[-k](population[i, 0:len(min_values)])
    return population

############################################################################

# Function: Offspring
def breeding(population, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [f1, f2, f3], size = 5):
    offspring   = np.zeros((size, population.shape[1]))
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0  
    for i in range (0, offspring.shape[0]):
        if (len(population) - 1 >= 3):
            i1, i2 = random.sample(range(0, len(population) - 1), 2)
        elif (len(population) - 1 == 0):
            i1 = 0
            i2 = 0
        else:
            i1 = 0
            i2 = 1
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_1 = i1
            parent_2 = i2
        else:
            parent_1 = i2
            parent_2 = i1
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            rand_c = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            if (rand_c >= 0.5):
                offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            else:   
                offspring[i,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [f1, f2, f3]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

############################################################################

# 输入：目标函数数量、在每个目标维度上的划分数量
# 输出：所有参考点列表的numpy数组格式
def reference_points(M, p):
    # 递归生成所有可能的参考点
    def generator(r_points, M, Q, T, D):
        points = []
        if (D == M - 1):
            r_points[D] = Q / T
            points.append(r_points)
        elif (D != M - 1):
            for i in range(Q + 1):
                r_points[D] = i / T
                points.extend(generator(r_points.copy(), M, Q - i, T, D + 1))
        return points
    ref_points = np.array(generator(np.zeros(M), M, p, p, 0))
    return ref_points

# Function: Normalize Objective Functions
# def normalization(population, number_of_functions):
#     M                 = number_of_functions
#     z_min             = np.min(population[:,-M:], axis = 0)
#     population[:,-M:] = population[:,-M:] - z_min
#     w                 = np.zeros((M, M)) + 0.0000001
#     np.fill_diagonal(w, 1)
#     z_max             = []
#     for i in range(0, M):
#        z_max.append(np.argmin(np.max(population[:,-M:]/w[i], axis = 1)))
#     if ( len(z_max) != len(set(z_max)) or M == 1):
#         a     = np.max(population[:,-M:], axis = 0)
#     else:
#         k     = np.ones((M, 1))
#         z_max = np.vstack((population[z_max,-M:]))
#         a     = np.matrix.dot(np.linalg.inv(z_max), k)
#         a     = (1/a).reshape(1, M)
#     population[:,-M:] = population[:,-M:] /(a - z_min)
#     return population

def normalization(population, number_of_functions):
    M = number_of_functions
    z_min = np.min(population[:, -M:], axis=0)
    population[:, -M:] = population[:, -M:] - z_min
    z_max = np.max(population[:, -M:], axis=0)

    # 逐元素缩放，而非求逆
    a = 1 / z_max  # 确保 z_max 中没有零值，否则这里会导致除以零的错误
    population[:, -M:] = population[:, -M:] * a  # 使用广播进行缩放

    return population


# Function: Distance from Point (p3) to a Line (p1, p2).    
def point_to_line(p1, p2, p3):
    p2 = p2 - p1
    dp = np.dot(p3, p2.T)
    pp = dp/np.linalg.norm(p2.T, axis = 0)
    pn = np.linalg.norm(p3, axis = 1)
    pn = np.array([pn,]*pp.shape[1]).transpose()
    dl = np.sqrt(pn**2 - pp**2)
    return dl

# 将种群每个个体与一组参考点（srp）关联起来，并基于这种关联选择出那些对超体积贡献最大的个体
def association(srp, population, z_max, number_of_functions):
    M    = number_of_functions
    p    = copy.deepcopy(population)
    p    = normalization(p, M)
    p1   = np.zeros((1, M))
    p2   = srp
    p3   = p[:,-M:]
    g    = point_to_line(p1, p2, p3) # Matrix (Population, Reference)
    idx  = []
    arg  = np.argmin(g, axis = 1)
    hv_c = pg.hypervolume(p[:,-M:])
    z    = np.max(p[:,-M:], axis = 0)
    if any(z > z_max):
        z_max = np.maximum(z_max,z)
    hv   = hv_c.contributions(z_max)
    d    = 1/(hv + 0.0000000000000001)
    for ind in np.unique(arg).tolist():
        f = [i[0] for i in np.argwhere(arg == ind).tolist()]
        idx.append(f[d[f].argsort()[0]])
    if (len(idx) < 5):   
        idx.extend([x for x in list(range(0, population.shape[0])) if x not in idx])
        idx = idx[:5]
    return idx

############################################################################

# 参考点的数量、变异率、决策变量的最小/大值、目标函数列表、迭代次数、控制交叉/变异的参数、与参考点相关的乘数，用来确定种群大小、是否打印详细消息
def hypervolume_estimation_mooa(references = 5, mutation_rate = 0.1, min_values = [0]*7, max_values = [1]*7, list_of_functions = [f1, f2, f3], generations = 921, mu = 1, eta = 1, k = 1, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    srp        = reference_points(M = M, p = references) # 参考点集合
    size       = k*srp.shape[0] # k * 参考点数量=1*21 =》种群大小
    population = initial_population(size, min_values, max_values, list_of_functions)  
    offspring  = initial_population(size, min_values, max_values, list_of_functions)  
    z_max      = np.max(population[:,-M:], axis = 0) # 一维数组：目标函数的最大值
    hypervolumes = []
    
    print('参考点总数量: ', int(srp.shape[0]), '种群大小', int(size))
    while (count <= generations):       
        # if (verbose == True):
        #     print('Generation = ', count)
        if count != 0:
            population = np.vstack([population, offspring]) # 将初始种群和后代种群合并
        z_max      = np.vstack([z_max, np.max(population[:,-M:], axis = 0)]) 
        z_max      = np.max(z_max, axis = 0) # 每一代都要更新z_max，它是目标函数值的历史最大值，用于计算超体积
        idx        = association(srp, population, z_max, M) # 得到对超体积贡献最大的个数
        population = population[idx, :]
        population = population[:size,:]

         # 计算并记录超体积
        hv = pg.hypervolume(population[:, -M:])
        hypervolumes.append(hv.compute(z_max))

        offspring  = breeding(population, min_values, max_values, mu, list_of_functions, size)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)             
        count      = count + 1              
    return population[:srp.shape[0], :], hypervolumes

def plot_hypervolumes(hypervolumes):
    plt.figure(figsize=(10, 5))
    plt.plot(hypervolumes, marker='o')
    plt.title('Hypervolume Progression Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.show()

def calculate_spacing(population):
    from scipy.spatial.distance import cdist
    # 计算所有解之间的距离
    distances = cdist(population, population, 'euclidean')
    # 设置对角线为无穷大，以避免选择自身为最近邻
    np.fill_diagonal(distances, np.inf)
    # 找到每个解的最近邻解距离
    nearest_distances = np.min(distances, axis=1)
    # 计算距离的平均值
    mean_distance = np.mean(nearest_distances)
    # 计算每个距离与平均值的差的平方
    spacing_value = np.sqrt(np.mean((nearest_distances - mean_distance) ** 2))
    return spacing_value

if __name__ == '__main__':
    start_time = time.time()
    populations, hypervolumes = hypervolume_estimation_mooa()
    spacing_values = calculate_spacing(populations[:, -3:])
    # 计算最高值、最低值和平均值
    elapsed_time = time.time() - start_time  # End timing
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    # plot_hypervolumes(hypervolumes)
    # max_hv = max(hypervolumes)
    # min_hv = min(hypervolumes)
    # average_hv = sum(hypervolumes) / len(hypervolumes)

    # print(f"最高HV值: {max_hv}")
    # print(f"最低HV值: {min_hv}")
    # print(f"平均HV值: {average_hv}")
    print("Spacing Value:", spacing_values)
    