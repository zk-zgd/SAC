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

# 快速非支配排序
def fast_non_dominated_sort(objectives):
    S = [[] for _ in range(len(objectives))]
    front = [[]]
    n = [0] * len(objectives)
    rank = [0] * len(objectives)

    for p in range(len(objectives)):
        S[p] = []
        n[p] = 0
        for q in range(len(objectives)):
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)
    
    return front

# 检查是否支配
def dominates(ind1, ind2):
    return all(x <= y for x, y in zip(ind1, ind2)) and any(x < y for x, y in zip(ind1, ind2))

def normalize_objectives(obj):
    min_vals = np.min(obj, axis=0)
    max_vals = np.max(obj, axis=0)
    
    # 检查最小值和最大值是否相等
    if np.any(min_vals == max_vals):
        raise ValueError("目标函数的最小值和最大值不能相等。")
    
    return (obj - min_vals) / (max_vals - min_vals)


# 超体积贡献计算
def calculate_hypervolume_contributions(front, reference_point):
    hv = pg.hypervolume(front)
    contributions = hv.contributions(reference_point)
    return contributions

# 更新种群（基于超体积贡献）
def update_population_with_hv(population, objectives, num_individuals, reference_point):
    all_contributions = []
    for front in fast_non_dominated_sort(objectives):
        if not front:
            continue
        front_values = [objectives[idx] for idx in front]
        
        # 只对非退化目标进行归一化
        try:
            nor_front = normalize_objectives(front_values)
        except ValueError:
            continue  # 跳过这个前沿
        contributions = calculate_hypervolume_contributions(nor_front, reference_point)
        for idx, contrib in zip(front, contributions):
            all_contributions.append((population[idx], objectives[idx], contrib))
    
    # 根据超体积贡献从大到小排序
    all_contributions.sort(key=lambda x: x[2], reverse=True)
    
    # 选择贡献最大的前num_individuals个体
    selected_individuals = all_contributions[:num_individuals]
    
    new_population = [ind[0] for ind in selected_individuals]
    new_objectives = [ind[1] for ind in selected_individuals]
    
    return new_population, new_objectives

toolbox.register("initialize", initialize_population, pop_size=100, num_nodes=N, num_shards=S)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=S, indpb=1/N)
toolbox.register("select", tools.selNSGA2)

# 保存数据到CSV
def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_fvmoea.csv"):
    data = {
        "Generation": range(1, len(hv_values) + 1),
        "Hypervolume": hv_values,
        "Throughput": throughput_values,
        "Delay": delay_values,
        "Security": security_values
    }
    df = pd.DataFrame(data)
    directory = "results-new-ref\\30010"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def calculate_hypervolume(solution_set, reference_point):
    for sol in solution_set:
        if sol[0] > 1:
            return
    hv = pg.hypervolume(solution_set)
    return hv.compute(reference_point)   

def fv_moea(pop_size, num_generations, num_nodes, num_shards):
    # 初始化种群并评估初始适应度
    population = initialize_population(pop_size, num_nodes, num_shards)
    for ind in population:
        ind.fitness.values = evaluate(ind)

    hv_values = []
    throughput_values = []
    delay_values = []
    security_values = []

    for gen in range(num_generations):
        if gen % 100 == 0:
            print("Generation:", gen)

        offspring = []
        
        # 生成子代，通过交叉和变异操作
        while len(offspring) < pop_size:
            parents = random.sample(population, 2)  # 选择父代
            
            # 默认复制父代作为子代
            child1 = parents[0].copy()
            child2 = parents[1].copy()
            
            if random.random() < 0.9:
                child1, child2 = tools.cxTwoPoint(parents[0], parents[1])  # 交叉操作

            # 变异操作
            tools.mutUniformInt(child1, 1, num_shards, indpb=1/N)
            tools.mutUniformInt(child2, 1, num_shards, indpb=1/N)
            # 确保子代为 creator.Individual 类型
            child1 = creator.Individual(child1)
            child2 = creator.Individual(child2)

            offspring.extend([child1, child2])


        # 评估子代的适应度
        for ind in offspring:
            ind.fitness.values = evaluate(ind)

        # 将子代加入到种群中
        population.extend(offspring)

        # 获取当前种群的非支配前沿
        # front = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
        # front_fitness = np.array([ind.fitness.values for ind in front])
        front_fitness = np.array([ind.fitness.values for ind in population])

        # 使用归一化后的目标函数值计算最大值
        normalized_front = normalize_objectives(front_fitness)
        reference_point = np.array([
            np.max(normalized_front[:, 0]) + 0.1,  # 吞吐量的参考点 (确保为最小化)
            np.max(normalized_front[:, 1]) + 0.1,  # 延迟的参考点
            np.max(normalized_front[:, 2]) + 0.1   # 安全性的参考点
        ])
        
        # 检查参考点是否有效
        if np.any(reference_point <= np.max(normalized_front, axis=0)):
            raise ValueError("参考点必须在非支配前沿的所有点之外，请调整参考点设置。")

        # 计算超体积
        hypervolume = calculate_hypervolume(normalized_front, reference_point)
        hv_values.append(hypervolume)

        # 计算当前种群的平均吞吐量、延迟和安全性
        avg_throughput = np.mean([-ind.fitness.values[0] for ind in population])  # 吞吐量取负值
        avg_delay = np.mean([ind.fitness.values[1] for ind in population])  # 延迟为正值
        avg_security = np.mean([ind.fitness.values[2] for ind in population])  # 安全性为正值

        # 将这些值存储到相应的列表中
        throughput_values.append(avg_throughput)
        delay_values.append(avg_delay)
        security_values.append(avg_security)

        # 计算超体积并更新种群
        population, _ = update_population_with_hv(population, normalized_front, pop_size, reference_point)

    return hv_values, throughput_values, delay_values, security_values

def main():
    pop_size = 100
    num_generations = 521
    num_nodes = N
    num_shards = S

    # 调用FV-MOEA算法并获得超体积、吞吐量、延迟和安全性的数据
    hv_values, throughput_values, delay_values, security_values = fv_moea(pop_size, num_generations, num_nodes, num_shards)

    # 保存所有数据到CSV文件
    save_data_to_csv(hv_values, throughput_values, delay_values, security_values)

    # 绘制结果
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 绘制HV变化图
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
    axs[1, 1].set_title('The std of malicious nodes Over Generations')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Security')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()