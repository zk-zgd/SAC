import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmo as pg  # 用于计算超体积
from deap import base, creator, tools
from scipy.spatial.distance import euclidean

# 定义适应度和个体
creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0, 1.0))  # 最大化吞吐量，最小化延迟和安全性
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# 属性生成器：在1到S之间的整数，表示节点所属的分片
toolbox.register("attr_int", random.randint, 1, 5)
# 个体初始化器
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=100)
# 种群初始化器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 从JSON文件加载节点信息
def load_node_info(file_path="E:\\mychain\\moa\\ZDT_MOA\\MaOEA\\NodeInfo2.json", encoding='utf-8'):
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
    for k, v in node_info.items():
        shard_id = int(k[:2])
        node_id = int(k[2:])
        node_global_id = (shard_id - 1) * 10 + node_id  # 假设每个分片有20个节点
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
def get_node_efficiencies(node_shard_history, shard_efficiencies):
    node_efficiencies = [0] * 100  # 假设有100个节点
    for shard_id, node_list in node_shard_history.items():
        for node_id in node_list:
            global_node_id = (shard_id - 1) * 10 + node_id  # 映射到全局节点ID
            node_efficiencies[global_node_id] = shard_efficiencies[shard_id - 1]
    return node_efficiencies

# 参数设置
N = 100  # 节点数
S = 10    # 分片个数
shard_efficiencies = [529, 666, 221, 851, 797, 745, 621, 456, 592, 664]  # 每个分片的效率

# 加载节点信息并初始化相关数据
node_info = load_node_info()
node_shard_history = get_node_shard_history(node_info)
distances = get_node_coordinates(node_info)
node_efficiencies = get_node_efficiencies(node_shard_history, shard_efficiencies)
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
    return average_delay  # 负值表示我们希望最小化延迟

# 目标函数3：安全性
def shard_security(individual):
    nodes_per_shard = get_nodes_per_shard(individual, S)
    malicious_counts = []
    for nodes in nodes_per_shard.values():
        malicious_nodes = sum(1 for node in nodes if reputations[node] == '恶意')  # '恶意'表示节点是恶意的
        malicious_counts.append(malicious_nodes)
    malicious_std = np.std(malicious_counts) if malicious_counts else 0
    return malicious_std  # 负值表示我们希望最小化标准差

# 评估函数
def evaluate(individual):
    throughput_val = throughput(individual)
    delay_val = delay(individual)
    security_val = shard_security(individual)
    return (throughput_val, delay_val, security_val)

# 初始化种群，确保每个分片至少有8个节点
def initialize_population(pop_size, num_nodes, num_shards):
    population = []
    for _ in range(pop_size):
        while True:
            individual_data = [random.randint(1, num_shards) for _ in range(num_nodes)]
            nodes_per_shard = get_nodes_per_shard(individual_data, num_shards)
            if all(len(nodes) >= 8 for nodes in nodes_per_shard.values()):
                break
        individual = creator.Individual(individual_data)
        population.append(individual)
    return population

# 注册工具箱函数
toolbox.register("initialize", initialize_population, pop_size=100, num_nodes=N, num_shards=S)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=S, indpb=1.0/N)

# 生成权重向量
def initialize_weights(N, M):
    weights = np.random.rand(N, M)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights

# 计算邻居
def calculate_neighbors(weights, T):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :T]
    return neighbors

# 标量化函数
def scalarizing_function(fitness_values, weight_vector, ideal_point):
    return np.max(weight_vector * np.abs(fitness_values - ideal_point))

# 保存数据到CSV文件
def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_moead.csv"):
    data = {
        "Generation": range(1, len(hv_values) + 1),
        "Hypervolume": hv_values,
        "Throughput": throughput_values,
        "Delay": delay_values,
        "Security": security_values
    }
    df = pd.DataFrame(data)
    directory = "results-new-ref/10"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def calculate_hypervolume(solution_set, reference_point):
    for sol in solution_set:
        if sol[0] > 1:
            return
    if np.any(np.isnan(solution_set)):
        print("Solution Set:", solution_set)
        raise ValueError("目标函数值包含NaN。")
    hv = pg.hypervolume(solution_set)
    return hv.compute(reference_point)     

def normalize_objectives(obj):
    min_vals = np.min(obj, axis=0)
    max_vals = np.max(obj, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (obj - min_vals) / range_vals
    # if np.any(np.isnan(obj)):
    #     print("Solution Set:", obj)
    #     raise ValueError("目标函数值包含NaN。")
    # return (obj - min_vals) / (max_vals - min_vals)

# 主函数
def main(pop_size=100, iter=521, cxpb=0.9, mutpb=0.1):
    # 初始化种群
    population = toolbox.initialize()
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    nobj = 3  # 目标函数数量
    T = 10     # 邻域大小
    
    # 生成权重向量和邻居
    weights = initialize_weights(pop_size, nobj)
    neighbors = calculate_neighbors(weights, T)
    
    # 初始化理想点
    ideal_point = np.array([
        min(population, key=lambda x: x.fitness.values[0]).fitness.values[0],  # 最小（最负）吞吐量
        min(population, key=lambda x: x.fitness.values[1]).fitness.values[1],  # 最小时延
        min(population, key=lambda x: x.fitness.values[2]).fitness.values[2]   # 最小标准差
    ])

    # 初始化用于绘图和保存的数据列表
    hv_values = []
    throughput_values = []
    delay_values = []
    security_values = []
    
    for gen in range(iter):
        if gen % 100 == 0:
            print(f"Generation {gen+1}/{iter}")
        for i, individual in enumerate(population):
            # 从邻域中选择父代
            mating_pool = [population[j] for j in neighbors[i]]
            parents = random.sample(mating_pool, 2)
            # 生成子代
            offspring = toolbox.clone(individual)
            # 执行交叉操作（根据交叉率）
            if random.random() < cxpb:
                offspring = toolbox.mate(parents[0], parents[1])[0]
            else:
                offspring = toolbox.clone(parents[0])
            
            # 执行变异操作（根据变异率）
            if random.random() < mutpb:
                offspring = toolbox.mutate(offspring)[0]
            
            del offspring.fitness.values            
            # 评估子代
            offspring.fitness.values = toolbox.evaluate(offspring)
            # 更新理想点
            ideal_point = np.minimum(ideal_point, offspring.fitness.values)
            # 更新邻居
            for idx in neighbors[i]:
                neighbor = population[idx]
                aggregation_neighbor = scalarizing_function(neighbor.fitness.values, weights[idx], ideal_point)
                aggregation_offspring = scalarizing_function(offspring.fitness.values, weights[idx], ideal_point)
                if aggregation_offspring < aggregation_neighbor:
                    population[idx] = toolbox.clone(offspring)
                    population[idx].fitness.values = offspring.fitness.values
        
        # 计算Hypervolume
        obj = np.array([ind.fitness.values for ind in population])
        obj_transformed = np.copy(obj)
        obj_transformed[:, 0] = -obj_transformed[:, 0]  # 将吞吐量最大化转为最小化

        # 归一化目标值
        obj_normalized = normalize_objectives(obj_transformed)

        # 更新参考点
        reference_point = np.max(obj_normalized, axis=0) + 0.1

        hypervolume = calculate_hypervolume(obj_normalized, reference_point)
        hv_values.append(hypervolume)

        # 记录指标
        throughput_mean = np.mean([ind.fitness.values[0] for ind in population])
        delay_mean = np.mean([ind.fitness.values[1] for ind in population])
        security_mean = np.mean([ind.fitness.values[2] for ind in population])

        throughput_values.append(throughput_mean)
        delay_values.append(delay_mean)
        security_values.append(security_mean)
    
    # 保存数据到CSV文件
    save_data_to_csv(hv_values, throughput_values, delay_values, security_values)
    
    # 绘制结果
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # 绘制超体积变化图
    axs[0, 0].plot(hv_values, marker='o', linestyle='-', color='b')
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
    axs[1, 1].set_title('Security Over Generations')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Security')
    axs[1, 1].grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
