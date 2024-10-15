import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmo as pg  # 用于计算超体积
from deap import base, creator, tools
from scipy.spatial.distance import euclidean, cdist
import networkx as nx


# 参数设置
N = 200  # 节点数
S = 10    # 分片个数
shard_efficiencies = [529, 666, 221, 851, 797, 745, 621, 456, 592, 664]  # 每个分片的效率

# 定义适应度和个体
creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0, 1.0))  # 最大化吞吐量，最小化延迟和安全性
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# 属性生成器：在1到S之间的整数，表示节点所属的分片
toolbox.register("attr_int", random.randint, 1, S)

# 从JSON文件加载节点信息
def load_node_info(file_path="E:\\mychain\\moa\\ZDT_MOA\\MaOEA\\NodeInfo3.json", encoding='utf-8'):
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
        node_global_id = (shard_id - 1) * 30 + node_id  # 假设每个分片有20个节点
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
    node_efficiencies = [0] * 200  # 假设有100个节点
    for shard_id, node_list in node_shard_history.items():
        for node_id in node_list:
            global_node_id = (shard_id - 1) * 20 + node_id  # 映射到全局节点ID
            node_efficiencies[global_node_id] = shard_efficiencies[shard_id - 1]
    return node_efficiencies


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
            if all(len(nodes) >= 12 for nodes in nodes_per_shard.values()):
                break
        individual = creator.Individual(individual_data)
        population.append(individual)
    return population

def update_reference_points(V, objs, association, nobj):
    centroid = np.mean(objs, axis=0)
    for i in range(len(V)):
        if np.sum(association == i) < (len(objs) / len(V)):
            direction = centroid - V[i]
            V[i] += 0.1 * direction / np.linalg.norm(direction)
    return V

def normalize_vectors(V):
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V /= norms
    return V

def calculate_theta(V):
    cosine_matrix = np.dot(V, V.T)
    norms = np.linalg.norm(V, axis=1)
    cosine_matrix /= norms[:, None]
    cosine_matrix /= norms[None, :]
    angles = np.arccos(np.clip(cosine_matrix, -1.0, 1.0))
    np.fill_diagonal(angles, np.pi)
    return np.min(angles, axis=0)

def reference_points(npop, nobj):
    return np.random.dirichlet(np.ones(nobj), size=npop)

def selection(pop, npop, nvar):
    pop = np.array(pop)  # 确保 pop 是一个 NumPy 数组
    ind = np.random.randint(0, pop.shape[0], npop)
    mating_pool = pop[ind]
    if npop % 2 == 1:
        mating_pool = np.concatenate((mating_pool, mating_pool[0].reshape(1, nvar)), axis=0)
    return mating_pool

def crossover(mating_pool, crossover_rate=0.9, num_shards=5):
    noff = len(mating_pool)
    dim = len(mating_pool[0])
    offspring = np.empty_like(mating_pool)
    for i in range(0, noff, 2):
        if np.random.rand() < crossover_rate and i + 1 < noff:
            # 常规单点交叉操作
            crossover_point = np.random.randint(1, dim)
            child1 = np.concatenate((mating_pool[i][:crossover_point], mating_pool[i+1][crossover_point:]))
            child2 = np.concatenate((mating_pool[i+1][:crossover_point], mating_pool[i][crossover_point:]))
            offspring[i], offspring[i+1] = child1, child2
        else:
            offspring[i], offspring[i+1] = mating_pool[i], mating_pool[i+1]

    # 确保每个分片至少有一个节点
    offspring = enforce_shard_presence(offspring, num_shards)
    return offspring

def enforce_shard_presence(offspring, num_shards):
    # 确保每个子代都至少包含每个分片的一个节点
    corrected_offspring = offspring.copy()
    for idx, child in enumerate(offspring):
        for shard in range(1, num_shards+1):
            if shard not in child:
                # 选择一个随机节点更改到缺失的分片
                replace_idx = np.random.choice(np.where(child != shard)[0])
                corrected_offspring[idx][replace_idx] = shard
    return corrected_offspring

def mutation(pop, mutation_rate=0.1, num_shards=S):
    npop, nvar = pop.shape
    for i in range(npop):
        for j in range(nvar):
            if np.random.rand() < mutation_rate:
                possible_shards = list(range(1, num_shards+1))
                current_shard = pop[i, j]
                # 确保不会从单一分片中移除唯一的节点
                if np.count_nonzero(pop[i] == current_shard) > 1:
                    possible_shards.remove(current_shard)
                new_shard = np.random.choice(possible_shards)
                pop[i, j] = new_shard
    return pop

def environmental_selection(population, objs, V, t, iter, alpha, nvar, nobj, theta, npop):
    original_objs = np.copy(objs)
    t_objs = objs - np.min(objs, axis=0)
    
    # 计算角度并找到最近的参考点
    angle = np.arccos(1 - cdist(t_objs, V, 'cosine'))
    association = np.argmin(angle, axis=1)
    theta0 = (t / iter) ** alpha

    next_pop = []
    selected_count = 0
    for i in range(np.unique(association).shape[0]):
        ind = np.where(association == i)[0]
        if len(ind) > 0:
            APD = (1 + nobj * theta0 * angle[ind, i] / (theta[i] + 1e-6)) * np.sqrt(np.sum(t_objs[ind] ** 2, axis=1))
            best = ind[np.argmin(APD)]
            next_pop.append(population[best])
            selected_count += 1

    # 如果仍然不足 npop 个体，随机补充
    while selected_count < npop:
        random_index = np.random.randint(0, len(population))
        next_pop.append(population[random_index])
        selected_count += 1

    return next_pop

def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_rvea.csv"):
    data = {
        "Generation": range(1, len(hv_values) + 1),
        "Hypervolume": hv_values,
        "Throughput": throughput_values,
        "Delay": delay_values,
        "Security": security_values
    }
    df = pd.DataFrame(data)
    directory = "results-new-ref\\20"
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


def main(npop=100, iter=521, nobj=3, eta_c=15, eta_m=20, alpha=0.1, fr=0.1):
    # 参数设置
    cxpb = 0.9
    mutpb = 0.1
    nvar = 200 
    # 注册工具箱函数
    toolbox.register("initialize", initialize_population, pop_size=100, num_nodes=N, num_shards=S)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=S, indpb=1.0/N)
    
    population = initialize_population(npop, N, S)

    for individual in population:
        individual.fitness.values = evaluate(individual)
    
    V0 = reference_points(npop, nobj)
    V = normalize_vectors(V0)
    theta = calculate_theta(V)
    hv_values, throughput_values, delay_values, security_values = [], [], [], []
    
    for t in range(iter):
        if t%100 == 0:
            print("t:", t+1)
        # 生成子代
        offspring = []
        while len(offspring) < npop:
            parents = random.sample(population, 2)
            if random.random() < cxpb:
                offspring1, offspring2 = toolbox.mate(parents[0], parents[1])
                del offspring1.fitness.values
                del offspring2.fitness.values
                offspring.extend([offspring1, offspring2])
        
        # 变异
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 合并父代和子代种群
        combined_population = population + offspring
        func_obj = []
        # 再次计算适应度
        for ind in combined_population:
            ind.fitness.values = toolbox.evaluate(ind)
            func_obj.append(ind.fitness.values)
        
        pop = environmental_selection(combined_population, func_obj, V, t, iter, alpha, nvar, nobj, theta, npop)
        
        # 计算超体积
        obj = np.array([ind.fitness.values for ind in pop])
        obj_transformed = np.copy(obj)
        obj_transformed[:, 0] = -obj_transformed[:, 0]
        # 归一化目标值
        obj_normalized = normalize_objectives(obj_transformed)

        # 更新参考点
        reference_point = np.max(obj_normalized, axis=0) + 0.1
        hypervolume = calculate_hypervolume(obj_normalized, reference_point)
        hv_values.append(hypervolume)

        # 记录指标
        throughput_values.append(np.mean([ind.fitness.values[0] for ind in population]))
        delay_values.append(np.mean([ind.fitness.values[1] for ind in population]))
        security_values.append(np.mean([ind.fitness.values[2] for ind in population]))
    save_data_to_csv(hv_values, throughput_values, delay_values, security_values)
    plot_metrics(hv_values, throughput_values, delay_values, security_values)

# 绘制图表
def plot_metrics(hv_values, throughput_values, delay_values, security_values):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
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

if __name__ == '__main__':
    main()


