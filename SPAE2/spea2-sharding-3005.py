import copy
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
S = 5    # 分片个数
shard_efficiencies = [529, 666, 221, 851, 797]  # 每个分片的效率
node_in_shard = [66, 57, 53, 64, 60]

toolbox = base.Toolbox()
# 属性生成器：在1到S之间的整数，表示节点所属的分片
toolbox.register("attr_int", random.randint, 1, S)
# 个体初始化器
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N)
# 种群初始化器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 从JSON文件加载节点信息
def load_node_info(file_path="config\\NodeInfo3005.json", encoding='utf-8'):
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
            if all(len(nodes) >= 40 for nodes in nodes_per_shard.values()):
                break
        individual = creator.Individual(individual_data)
        population.append(individual)
    return population

def calculate_hypervolume(solution_set, reference_point):
    for sol in solution_set:
        if sol[0] > 1:
            return
    hv = pg.hypervolume(solution_set)
    return hv.compute(reference_point)        

def create_plot(hv_values, throughput_values, delay_values, security_values):
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

def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_spea2.csv"):
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
    directory = "results-new-ref\\3005"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    
    # 保存到CSV
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def normalize_objectives(obj):
    min_vals = np.min(obj, axis=0)
    max_vals = np.max(obj, axis=0)
    return (obj - min_vals) / (max_vals - min_vals)

def main():
    # 参数设置
    pop_size = 100
    num_nodes = N
    num_shards = S
    generations = 521

    # 注册遗传算法的操作
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=S, indpb=0.05)
    toolbox.register("select", tools.selSPEA2)

    # 初始化种群
    population = initialize_population(pop_size, num_nodes, num_shards)

    # 评估初始种群
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 存储各代的指标
    hypervolumes = []
    throughput_values = []
    delay_values = []
    security_values = []

    for gen in range(generations):
        if gen % 100 == 0:
            print(f"Generation {gen+1}/{generations}")

        # 选择
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < 0.1:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 合并种群和子代
        population = population + offspring

        # 选择下一代
        population = toolbox.select(population, pop_size)

        # 计算Hypervolume
        obj = np.array([ind.fitness.values for ind in population])
        obj_transformed = np.copy(obj)
        obj_transformed[:, 0] = -obj_transformed[:, 0]  # 将吞吐量最大化转为最小化

        # 归一化目标值
        obj_normalized = normalize_objectives(obj_transformed)

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

    # 获取归档
    archive = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]

    # 计算最终Hypervolume
    obj_final = np.array([ind.fitness.values for ind in archive])
    obj_final_transformed = np.copy(obj_final)
    obj_final_transformed[:,0] = -obj_final_transformed[:,0]  # 将吞吐量最大化转为最小化
    final_hv = calculate_hypervolume(obj_final_transformed, reference_point)

    print(f"Final Hypervolume: {final_hv}")
    print(f"Number of Pareto-optimal solutions: {len(archive)}")

    # 绘制结果
    create_plot(hypervolumes, throughput_values, delay_values, security_values)
    save_data_to_csv(hypervolumes, throughput_values, delay_values, security_values)

if __name__ == "__main__":
    main()
