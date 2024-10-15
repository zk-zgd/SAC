import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import pygmo as pg
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import euclidean
import json

# 定义适应度和个体
creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0, 1.0))  # 最大化吞吐量，最小化延迟和安全性
creator.create("Individual", list, fitness=creator.FitnessMax)

# 参数设置
N = 300  # 节点数
S = 15    # 分片个数
shard_efficiencies = [529, 666, 221, 851, 797, 745, 621, 456, 592, 664, 431, 653, 443, 556, 798]   # 每个分片的效率
node_in_shard = [22, 21, 19, 18, 25, 16, 22, 23, 21, 17, 19, 19, 15, 20, 23]

toolbox = base.Toolbox()
# 属性生成器：在1到S之间的整数，表示节点所属的分片
toolbox.register("attr_int", random.randint, 1, S)
# 个体初始化器
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N)
# 种群初始化器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 从JSON文件加载节点信息
def load_node_info(file_path="config\\NodeInfo30015.json", encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        node_info = json.load(f)
    return node_info

# 获取节点所属的历史分片信息
def get_node_shard_history(node_info):
    node_shard_history = {}
    for key, details in node_info.items():
        shard_id = int(key[:2])  # 提取前两位作为分片ID
        node_id = int(key[2:])   # 提取后两位作为节点ID
        if shard_id not in node_shard_history:
            node_shard_history[shard_id] = []
        node_shard_history[shard_id].append(node_id)
    return node_shard_history

# 获取节点的信誉值
def get_node_reputation(node_info):
    node_reputation = []
    for v in node_info.values():
        reputation = v['reputation']
        node_reputation.append(reputation)
    return node_reputation

# 获取节点的坐标并计算节点之间的距离
def get_node_coordinates(node_info):
    coordinates = {}
    for n in node_in_shard:
        for k, v in node_info.items():
            shard_id = int(k[:2])
            node_id = int(k[2:])
            node_global_id = shard_id * n + node_id  
            coordinates[node_global_id] = tuple(v['coordinates'])
    distances = {}
    node_ids = list(coordinates.keys())
    for i in node_ids:
        for j in node_ids:
            if i != j:
                if (i, j) not in distances and (j, i) not in distances:
                    distances[(i, j)] = round(euclidean(coordinates[i], coordinates[j]), 2)
    return distances

# 获取节点的处理效率
def get_node_efficiencies(node_shard_history, shard_efficiencies, node_in_shard):
    node_efficiencies = [0] * N  
    # 计算每个分片之前的节点数总和，用于全局ID的计算
    shard_offset = [0] * len(node_in_shard)
    for i in range(1, len(node_in_shard)):
        shard_offset[i] = shard_offset[i - 1] + node_in_shard[i - 1]

    # 分片历史中的节点ID对应其全局ID，并赋予效率
    for shard_id, node_list in node_shard_history.items():
        for node_id in node_list:
            global_node_id = shard_offset[shard_id] + node_id  # 计算全局节点ID
            node_efficiencies[global_node_id] = shard_efficiencies[shard_id]
    
    return node_efficiencies

# 加载节点信息并初始化相关数据
node_info = load_node_info()
node_shard_history = get_node_shard_history(node_info)
distances = get_node_coordinates(node_info)
node_efficiencies = get_node_efficiencies(node_shard_history, shard_efficiencies, node_in_shard)
reputations = get_node_reputation(node_info)

# 辅助函数：获取每个分片的节点列表
def get_nodes_per_shard(individual, shard_count):
    nodes_per_shard = {shard: [] for shard in range(1, shard_count + 1)}
    for node_index, shard_index in enumerate(individual):
        if shard_index in nodes_per_shard:
            nodes_per_shard[shard_index].append(node_index)
        else:
            print("Invalid shard index:", shard_index)
    return nodes_per_shard

# 目标函数1：吞吐量
def throughput(individual):
    nodes_per_shard = get_nodes_per_shard(individual, S)
    total_throughput = 0
    for nodes in nodes_per_shard.values():
        if nodes:
            efficiencies = sum(node_efficiencies[node] for node in nodes)
            shard_throughput = efficiencies / len(nodes)
            total_throughput += shard_throughput
    return -total_throughput

# 目标函数2：延迟
def delay(individual):
    nodes_per_shard = get_nodes_per_shard(individual, S)
    total_delay = 0
    pair_count = 0
    for nodes in nodes_per_shard.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                distance = distances.get((node_i, node_j), distances.get((node_j, node_i), 0))
                total_delay += distance
                pair_count += 1
    average_delay = total_delay / pair_count if pair_count > 0 else 0
    return average_delay  

# 目标函数3：安全性
def shard_security(individual):
    nodes_per_shard = get_nodes_per_shard(individual, S)
    malicious_counts = []
    for nodes in nodes_per_shard.values():
        malicious_nodes = sum(1 for node in nodes if reputations[node] == '恶意')  # '恶意'表示节点是恶意的
        malicious_counts.append(malicious_nodes)
    malicious_std = np.std(malicious_counts) if malicious_counts else 0
    return malicious_std  

# 评估函数
def evaluate(individual):
    throughput_val = throughput(individual)
    delay_val = delay(individual)
    security_val = shard_security(individual)
    return (throughput_val, delay_val, security_val)

# 初始化种群，确保每个分片至少有10个节点
def initialize_population(pop_size, num_nodes, num_shards):
    population = []
    for _ in range(pop_size):
        while True:
            individual_data = [random.randint(1, num_shards) for _ in range(num_nodes)]
            nodes_per_shard = get_nodes_per_shard(individual_data, num_shards)
            if all(len(nodes) >= 12 for nodes in nodes_per_shard.values()):
                break
        individual = creator.Individual(individual_data)
        population.append(individual)
    return population

############################################################################
def select_parents(population):
    """根据二元锦标赛选择两个父母"""
    idx1, idx2 = np.random.choice(range(len(population)), 2, replace=False)
    return population[idx1], population[idx2]

def sbx_crossover(parent1, parent2, prob_crossover=0.9, eta_c=15):
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < prob_crossover:
        min_len = min(len(parent1), len(parent2))
        for i in range(min_len):
            u = np.random.rand()
            beta = 2.0 ** (-1.0 / (eta_c + 1)) if u <= 0.5 else (2.0 * u) ** (1.0 / (eta_c + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 + beta) * parent2[i] + (1 - beta) * parent1[i])
            # 确保交叉后的值在合理范围内并转为整数
            child1[i] = int(np.clip(child1[i], 1, 5))
            child2[i] = int(np.clip(child2[i], 1, 5))
    return child1, child2

def polynomial_mutation(child, prob_mutation=0.1, eta_m=20):
    for node in child:
        if np.random.rand() < prob_mutation:
            u = np.random.rand()
            delta = (2 * u - 1) if u < 0.5 else (1 - 2 * (1 - u))
            delta_q = delta ** (1 / (eta_m + 1))
            node = node + delta_q * (node - 1)  # Assume `1` is the lower bound
            node = int(np.clip(node, 1, 5))  # 转为整数并确保在范围内
    return child

############################################################################

# Function: Reference Points
def reference_points(M, p):
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

# 将多个目标函数值归一化，以便他们可以在一个统一的比较基础上被评估
def normalization(function_values):
    # 将 function_values 转换为 NumPy 数组并转置
    values_matrix = np.array(function_values)

    # 计算每一列的最小值和最大值
    z_min = np.min(values_matrix, axis=0)
    z_max = np.max(values_matrix, axis=0)

    # 创建一个布尔数组，标识每一列 z_max - z_min 是否为非零
    non_zero_mask = (z_max - z_min) != 0

    # 初始化归一化矩阵，形状与 values_matrix 相同
    normalized_matrix = np.zeros_like(values_matrix)

    # 对于非零差值的列进行归一化处理
    normalized_matrix[:, non_zero_mask] = (
        (values_matrix[:, non_zero_mask] - z_min[non_zero_mask]) / 
        (z_max[non_zero_mask] - z_min[non_zero_mask])
    )

    # print("normalized_matrix shape:", normalized_matrix.shape)
    # 将归一化后的函数值存入对应的个体中
    # for i, ind in enumerate(population):
    #     population[i] = np.append(population[i])

    return normalized_matrix

def calculate_hypervolume(solution_set, reference_point):
    for sol in solution_set:
        if sol[0] > 1:
            return
    hv = pg.hypervolume(solution_set)
    return hv.compute(reference_point)     

def point_to_line(p1, p2, p3):
    # p1: 参考点集合, 形状 (num_reference_points, M)
    # p2: 个体的目标函数值集合, 形状 (num_individuals, M)
    # p3: 归一化后的目标函数值集合, 形状 (num_individuals, M)
    # 确保 p1, p2, p3 是 NumPy 数组
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    # 将p1的形状扩展为 (1, num_reference_points, M) 以便后续广播运算
    p1 = np.expand_dims(p1, axis=0)  # 形状变为 (1, num_reference_points, M)
    
    # 计算p3到p1的向量差，结果形状为 (num_individuals, num_reference_points, M)
    vectors_diff = p3[:, np.newaxis, :] - p1  
    
    # 归一化p2，用于投影计算
    norm_p2 = np.linalg.norm(p2, axis=1, keepdims=True)  # 形状: (num_individuals, 1)
    norm_p2[norm_p2 == 0] = 1e-10  # 避免除以零
    p2_normalized = p2 / norm_p2  # 形状: (num_individuals, M)
    
    # 调整p2的形状用于点积运算
    p2_normalized = np.expand_dims(p2_normalized, axis=1)  # 形状: (num_individuals, 1, M)
    
    # 计算p3在p2上的投影长度
    projection_lengths = np.sum(vectors_diff * p2_normalized, axis=2)  # 形状: (num_individuals, num_reference_points)
    
    # 计算投影向量
    projection_vectors = p2_normalized * np.expand_dims(projection_lengths, axis=2)  # 形状: (num_individuals, num_reference_points, M)
    
    # 计算p3到参考点连线的垂直距离
    orthogonal_distances = np.linalg.norm(vectors_diff - projection_vectors, axis=2)  # 形状: (num_individuals, num_reference_points)
    
    return orthogonal_distances

toolbox.register("initialize", initialize_population, pop_size=100, num_nodes=N, num_shards=S)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", sbx_crossover)
toolbox.register("mutate", polynomial_mutation)
toolbox.register("select", select_parents)

# 将个体与参考点关联：如果某个参考点周围的个体较多，可能会选择距离最近的几个，较远的会被淘汰，以促进解的多样性
def association(srp, population, z_max, function_values):
    normalized_matrix = normalization(function_values)
    func_matrix = np.array(function_values)

    values_matrix = normalized_matrix

    if np.isnan(values_matrix).any() or np.isinf(values_matrix).any():
        raise ValueError("The values matrix contains NaN or Inf values, which are not allowed.")

    # print("srp shape:", srp.shape)
    # print("func_matrix shape:", np.shape(func_matrix))
    # print("values_matrix shape:", np.shape(values_matrix))
    p1 = np.array(srp)  # 参考点集合【21,3】
    p2 = func_matrix # 个体函数值集合 【159,3】
    p3 = values_matrix  # Normalized 个体函数值【159,3】

    g = point_to_line(p1, p2, p3)  # 计算每个个体到参考点的距离，并选择最小距离的索引
    idx = []
    arg = np.argmin(g, axis=1)
    
    try:
        # 计算HV贡献度
        hv_c = pg.hypervolume(values_matrix)
        z = np.max(values_matrix, axis=0)
        if any(z > z_max):
            z_max = np.maximum(z_max, z)
        
        hv = hv_c.contributions(z_max)
        d = 1 / (hv + 1e-10)
        
        for ind in np.unique(arg).tolist():
            f = [i[0] for i in np.argwhere(arg == ind).tolist()]
            idx.append(f[d[f].argsort()[0]])
        
        if len(idx) < 5:
            idx.extend([x for x in range(len(population)) if x not in idx])
            idx = idx[:5]
    except ValueError as e:
        print(f"Error calculating hypervolume: {e}")

    return idx
############################################################################
def hypervolume_estimation_mooa(references=5, list_of_functions=[throughput, delay, shard_security], generations=521, k=5):
    ##################
    # 1. 初始化种群，并对目标函数值归一化（归一化步骤将目标函数值缩放到一个相对范围内，以便在计算距离、关联度等时具有统一的尺度。）
    # 2. 计算每个解的超体积贡献，并将解与参考点进行关联
    # 3. 基于超体积贡献和关联度，选择解并进行交叉、变异生成下一代
    ##################
    count = 0
    references = max(5, references)
    M = len(list_of_functions)
    srp = reference_points(M, references) # 根据目标个数和参考点个数，生成参考点矩阵srp
    srp_size = srp.shape[0]
    size = k * srp_size # 计算参考点对应的种群大小
    population = initialize_population(size, N, S)
    func_values = []
    for ind in population:
        ind.fitness.values = evaluate(ind)
        func_values.append(ind.fitness.values)
    # 转换为NumPy数组，方便后续操作
    func_values = np.array(func_values)

    # 归一化处理
    normalized_func_values = normalization(func_values)

    # 计算每个目标的最大值，并添加0.0001作为偏移
    z_max = np.max(normalized_func_values, axis=0) + 0.0001

    hypervolumes = []
    throughput_values = []
    delay_values = []
    security_values = []    
    # 主循环，执行进化过程
    while count <= generations:
        if count % 100 ==0:
            print("gen=", count)
        func_values = []
        
        # 计算当前种群的适应度值
        if count != 0:
            population.extend(offspring)  # 扩展种群，添加后代
            for ind in population:
                ind.fitness.values = evaluate(ind)
                func_values.append(ind.fitness.values)
            
            # 转换为NumPy数组进行处理
            func_values = np.array(func_values)

            # 计算参考点z_max，进行归一化
            z_max = np.max(func_values, axis=0) + 0.0001  # 保证每个维度的最大值+偏移量

            # 关联性操作，选择合适的种群个体
            idx = association(srp, population, z_max, func_values)
            population = [population[i] for i in idx][:size]  # 保持种群大小为 size
        
        # 计算Hypervolume
        obj = np.array([ind.fitness.values for ind in population])
        obj_transformed = np.copy(obj)
        obj_transformed[:, 0] = -obj_transformed[:, 0]  # 将吞吐量最大化转为最小化

        # 归一化目标值
        obj_normalized = normalization(obj_transformed)

        # 更新参考点
        reference_point = np.max(obj_normalized, axis=0) + 0.1

        hypervolume = calculate_hypervolume(obj_normalized, reference_point)
        hypervolumes.append(hypervolume)

        # 记录指标
        throughput_mean = np.mean([ind.fitness.values[0] for ind in population])
        delay_mean = np.mean([ind.fitness.values[1] for ind in population])
        security_mean = np.mean([ind.fitness.values[2] for ind in population])

        throughput_values.append(throughput_mean)
        delay_values.append(delay_mean)
        security_values.append(security_mean)
        # 生成后代
        offspring = []
        while len(offspring) < 100:
            parents = random.sample(population, 2)  # 选择父代
            child1 = parents[0].copy()
            child2 = parents[1].copy()
            if random.random() < 0.9 :
                child1, child2 = toolbox.mate(parents[0], parents[1])  
            toolbox.mutate(child1)
            toolbox.mutate(child2)
            child1 = creator.Individual(child1)
            child2 = creator.Individual(child2)
            offspring.extend([child1, child2])

        count += 1

    # 返回最终种群和评估的超体积、吞吐量、延迟和安全性
    return population, hypervolumes, throughput_values, delay_values, security_values

############################################################################
def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_hype.csv"):
    # 创建DataFrame
    data = {
        "Generation": range(1, len(hv_values) + 1),
        "Hypervolume": hv_values,
        "Throughput": throughput_values,
        "Delay": delay_values,
        "Security": security_values
    }
    df = pd.DataFrame(data)
    
    # 指定保存路径，这里假设在当前目录下创建一个results文件夹存储结果
    directory = "results-new-ref\\30015"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    
    # 保存到CSV
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

############################################################################

if __name__ == "__main__":
    # 调用 HypE 算法
    final_population, hypervolumes, throughput_values, delay_values, security_values = hypervolume_estimation_mooa()
    generations = range(len(hypervolumes))
    save_data_to_csv(hypervolumes, throughput_values, delay_values, security_values)
#################################################################################################
   
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 绘制HV变化图
    axs[0, 0].plot(hypervolumes, marker='o', linestyle='-', color='b')
    axs[0, 0].set_title('Hypervolume Over Generations')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Hypervolume')
    axs[0, 0].grid(True)

    # 绘制吞吐量变化图
    axs[0, 1].plot(throughput_values, marker='o', linestyle='-', color='g')
    axs[0, 1].set_title('Throughput Over Generations')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Throughput (tps)')
    axs[0, 1].grid(True)

    # 绘制延迟变化图
    axs[1, 0].plot(delay_values, marker='o', linestyle='-', color='m')
    axs[1, 0].set_title('Delay Over Generations')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('Delay')
    axs[1, 0].grid(True)

    # 绘制安全性变化图
    axs[1, 1].plot(security_values, marker='o', linestyle='-', color='r')
    axs[1, 1].set_title('The std of malicious nodes Over Generations')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Security')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()