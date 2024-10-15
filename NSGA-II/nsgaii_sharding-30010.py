import json
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmo as pg  # 用于计算超体积
from deap import base, creator, tools
from scipy.spatial.distance import euclidean
import networkx as nx

# 定义适应度和个体
creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0, 1.0))  # 最大化吞吐量，最小化延迟和安全性
creator.create("Individual", list, fitness=creator.FitnessMax)

# 参数设置
N = 300  # 节点数
S = 10    # 分片个数
shard_efficiencies = [529, 666, 221, 851, 797, 745, 621, 456, 592, 664]
node_in_shard = [30, 28, 41, 33, 24, 25, 25, 26, 38, 30]

toolbox = base.Toolbox()
# 属性生成器：在1到S之间的整数，表示节点所属的分片
toolbox.register("attr_int", random.randint, 1, S)
# 个体初始化器
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N)
# 种群初始化器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 从JSON文件加载节点信息
def load_node_info(file_path="config\\NodeInfo30010.json", encoding='utf-8'):
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
            if all(len(nodes) >= 15 for nodes in nodes_per_shard.values()):
                break
        individual = creator.Individual(individual_data)
        population.append(individual)
    return population


# 绘制图表
def plot_metrics(hv_values, throughput_values, delay_values, security_values):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(hv_values, marker='o', color='b')
    axs[0, 1].plot(throughput_values, marker='o', color='g')
    axs[1, 0].plot(delay_values, marker='o', color='m')
    axs[1, 1].plot(security_values, marker='o', color='r')
    plt.show()

# 保存数据到CSV文件
def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_nsgaii.csv"):
    data = {
        "Generation": range(1, len(hv_values) + 1),
        "Hypervolume": hv_values,
        "Throughput": throughput_values,
        "Delay": delay_values,
        "Security": security_values
    }
    df = pd.DataFrame(data)
    directory = "results-new-ref/30010"
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

def main():
    pop_size = 100
    max_gen = 521
    cxpb = 0.9  # 交叉率
    mutpb = 0.1  # 变异率

    # 注册工具箱函数
    toolbox.register("initialize", initialize_population, pop_size=pop_size, num_nodes=N, num_shards=S)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=S, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)
    population = toolbox.population(n=pop_size)
    hv_values = []
    throughput_values = []
    delay_values = []
    security_values = []
    gen_no = 0

    for gen_no in range(max_gen):
        if gen_no % 100 == 0:
            print("gen:", gen_no + 1)
        # 计算适应度
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # 非支配排序
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)

        # 计算拥挤距离
        for front in fronts:
            tools.emo.assignCrowdingDist(front)
        
        # 选择
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # 合并父代和子代种群
        combined_population = population + offspring

        # 再次进行非支配排序以选择新的种群
        combined_fronts = tools.sortNondominated(combined_population, len(combined_population))

        # 选择新的种群
        new_population = []
        for front in combined_fronts:
            new_population.extend(front)
            if len(new_population) >= pop_size:
                break
        
        population = new_population[:pop_size]

        # 记录指标
        throughput_values.append(np.mean([ind.fitness.values[0] for ind in population]))
        delay_values.append(np.mean([ind.fitness.values[1] for ind in population]))
        security_values.append(np.mean([ind.fitness.values[2] for ind in population]))

        # 计算超体积
        obj = np.array([ind.fitness.values for ind in population])
        obj_transformed = np.copy(obj)
        obj_transformed[:, 0] = -obj_transformed[:, 0]
        # 归一化目标值
        obj_normalized = normalize_objectives(obj_transformed)

        # 更新参考点
        reference_point = np.max(obj_normalized, axis=0) + 0.1
        hypervolume = calculate_hypervolume(obj_normalized, reference_point)
        hv_values.append(hypervolume)


        # 打印最后一代结果
        # if gen_no == max_gen:
        #     print(f"Generation {gen_no} Results:")
        #     print(f"Average Throughput: {np.mean(function1_values)}, Std Dev: {np.std(function1_values)}")
        #     print(f"Average Delay: {np.mean(function2_values)}, Std Dev: {np.std(function2_values)}")
        #     print(f"Average Security: {np.mean(function3_values)}, Std Dev: {np.std(function3_values)}")

    # 保存数据并绘制结果
    save_data_to_csv(hv_values, throughput_values, delay_values, security_values)
    plot_metrics(hv_values, throughput_values, delay_values, security_values)

    max_hv = max(hv_values)
    min_hv = min(hv_values)
    average_hv = sum(hv_values) / len(hv_values)

    print(f"最高HV值: {max_hv}")
    print(f"最低HV值: {min_hv}")
    print(f"平均HV值: {average_hv}")

if __name__ == "__main__":
    main()