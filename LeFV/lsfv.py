import random
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import time
import networkx as nx
import community as community_louvain
import pandas as pd
import os
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import euclidean
import json

# 从JSON文件加载节点信息
def load_node_info(file_path="E:\\mychain\\moa\\ZDT_MOA\\MaOEA\\NodeInfo.json", encoding='utf-8'):
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
        node_global_id = (shard_id - 1) * 20 + node_id  # 假设每个分片有20个节点
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
            global_node_id = (shard_id - 1) * 20 + node_id  # 映射到全局节点ID
            node_efficiencies[global_node_id] = shard_efficiencies[shard_id - 1]
    return node_efficiencies

# 参数设置
N = 100  # 节点数
S = 5    # 分片个数
shard_efficiencies = [529, 666, 221, 851, 797]  # 每个分片的效率

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

# 初始化种群，确保每个分片至少有12个节点
def initialize_population(pop_size, num_nodes, num_shards):
    population = []
    for _ in range(pop_size):
        while True:
            individual_data = [random.randint(1, num_shards) for _ in range(num_nodes)]
            nodes_per_shard = get_nodes_per_shard(individual_data, num_shards)
            if all(len(nodes) >= 12 for nodes in nodes_per_shard.values()):
                break
        # individual = creator.Individual(individual_data)
        population.append(individual_data)
    return population


# 分布指数eta_c越大，子代越趋向于父代
def sbx_crossover(parent1, parent2, prob_crossover, eta_c=15):
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # 随机决定是否进行交叉
    if np.random.rand() < prob_crossover:
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:  # 对每个基因位随机选择是否交叉
                # 计算交叉的beta参数
                u = np.random.rand()
                beta = 2.0 ** (-1.0 / (eta_c + 1)) if u <= 0.5 else (2.0 * u) ** (1.0 / (eta_c + 1))
                
                # 生成两个子代
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                
                # 确保子代仍在有效范围内（例如分片编号）
                child1[i] = min(max(int(round(child1[i])), 1), 5)
                child2[i] = min(max(int(round(child2[i])), 1), 5)
    
    return child1, child2

def polynomial_mutation(individual, prob_mutation, eta_m=20):
    """多项式变异"""    
    for i in range(len(individual)):
        if np.random.rand() < prob_mutation:
            u = np.random.rand()
            delta = np.where(u < 0.5, (2*u)**(1/(eta_m+1)) - 1, 1 - (2*(1-u))**(1/(eta_m+1)))
            individual[i] += delta
            individual[i] = int(np.clip(round(individual[i]), 1, S))
    return individual

def select_parents(population):
    """根据二元锦标赛选择两个父母"""
    idx1, idx2 = np.random.choice(range(len(population)), 2, replace=False)
    return population[idx1], population[idx2]

def fast_non_dominated_sort(values):
    """ 快速非支配排序 """
    S = [[] for _ in range(len(values))]
    front = [[]]
    n = [0] * len(values)
    rank = [0] * len(values)

    for p in range(len(values)):
        S[p] = []
        n[p] = 0
        for q in range(len(values)):
            if dominates(values[p], values[q]):
                S[p].append(q)
            elif dominates(values[q], values[p]):
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

def dominates(ind1, ind2):
    """ 检查个体1是否支配个体2 """
    return all(x <= y for x, y in zip(ind1, ind2)) and any(x < y for x, y in zip(ind1, ind2))

def handle_extreme_values(objectives, upper_bounds, lower_bounds):
    """ 将目标值限制在合理的范围内 """
    adjusted_objectives = []
    for obj, upper, lower in zip(objectives, upper_bounds, lower_bounds):
        if obj > upper:
            adjusted_obj = upper
        elif obj < lower:
            adjusted_obj = lower
        else:
            adjusted_obj = obj
        adjusted_objectives.append(adjusted_obj)
    return adjusted_objectives

def calculate_hypervolume_contributions(population, objectives, reference_point):
    all_contributions = []
    for front in fast_non_dominated_sort(objectives):
        if not front:
            continue
        front_values = [objectives[idx] for idx in front]
        hv = pg.hypervolume(front_values)
        contributions = hv.contributions(reference_point)
        for idx, contrib in zip(front, contributions):
            all_contributions.append((population[idx], objectives[idx], contrib))
    
    return all_contributions

# def update_population_with_hv(population, objectives, num_individuals, reference_point):
    
#     # 将150个目标函数值按快速非支配排序分开计算contris
#     all_contributions = calculate_hypervolume_contributions(population, objectives, reference_point)  
#     # Sort by hypervolume contributions
#     all_contributions.sort(key=lambda x: x[2], reverse=True)
    
#     # 取100行之前的，保留为新种群((个体，目标值)，目标值，贡献)
#     new_population = all_contributions[:num_individuals]
#     # 取100行之后的50个个体((个体，目标值)，目标值，贡献)
#     remained_individuals = all_contributions[num_individuals:]

#     # (个体, 目标值)
#     new_population_data = [ind for ind, obj, _ in new_population]
#     # 进行局部搜索优化以优化重复的个体 (个体，目标值)
#     remained_population = perform_local_search(remained_individuals)

#     # 结合新种群和优化后的剩余种群
#     new_population_data.extend(remained_population)
#     new_objectives = [obj for _, obj in new_population_data]
#     reference_point = np.max(new_objectives, axis=0) + 0.0001

#     # 使用快速非支配排序和超体积贡献来更新种群
#     all_contributions = calculate_hypervolume_contributions(new_population_data, new_objectives, reference_point)
#     # 根据超体积贡献进行排序
#     all_contributions.sort(key=lambda x: x[2], reverse=True)
#     new_population = [x[0] for x in all_contributions[:num_individuals]]
#     new_objectives = [x[1] for x in all_contributions[:num_individuals]]
    
#     return new_population, new_objectives

# def perform_local_search(remained_individuals, num_selected=50):
#     """使用拥挤距离选择优秀个体，只返回个体和目标值"""
#     # 提取个体和目标值
#     objectives = [obj[1] for obj in remained_individuals]

#     # 计算拥挤距离
#     crowding_distances = calculate_crowding_distance(objectives)

#     # 结合个体索引和拥挤距离，选择拥挤距离最大的50个个体
#     indexed_distances = list(enumerate(crowding_distances))
#     sorted_by_distance = sorted(indexed_distances, key=lambda x: x[1], reverse=True)
#     selected_indices = [idx for idx, dist in sorted_by_distance[:num_selected]]
    
#     # 提取选中的个体及其目标值
#     selected_individuals = [remained_individuals[i][0] for i in selected_indices]
#     selected_objectives = [remained_individuals[i][1] for i in selected_indices]

#     # 返回选中的个体及其目标值，不包含拥挤距离
#     return list(zip(selected_individuals, selected_objectives))

def calculate_crowding_distance(objectives):
    """计算每个个体的拥挤距离"""
    num_objectives = len(objectives[0])
    num_individuals = len(objectives)
    distances = np.zeros(num_individuals)
    
    # 对每个目标维度进行操作
    for i in range(num_objectives):
        sorted_indices = np.argsort([obj[i] for obj in objectives])
        max_value = objectives[sorted_indices[-1]][i]
        min_value = objectives[sorted_indices[0]][i]

        # 为边界个体设置无穷大的拥挤距离
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # 计算中间个体的拥挤距离
        if max_value - min_value != 0:  # 只有当最大值和最小值不相等时才执行
            for j in range(1, num_individuals - 1):
                distances[sorted_indices[j]] += (objectives[sorted_indices[j + 1]][i] - objectives[sorted_indices[j - 1]][i]) / (max_value - min_value)
        else:  # 当最大值等于最小值时，距离增量为0（或其他合适的值）
            for j in range(1, num_individuals - 1):
                distances[sorted_indices[j]] += 0
    
    return distances
def local_search(individual, mutation_rate):
    """简单的局部搜索策略，通过小范围的扰动来优化个体"""
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # 在邻域内随机选择一个新的分片
            new_shard = random.randint(1, S)
            individual[i] = new_shard
    return individual

def perform_local_search(population, num_selected=50, mutation_rate=0.1):
    """使用局部搜索优化选择的个体"""
    # 随机选择一部分个体进行局部搜索
    selected_indices = np.random.choice(range(len(population)), num_selected, replace=False)
    for idx in selected_indices:
        population[idx] = local_search(population[idx], mutation_rate)
    return population

def update_population_with_hv(population, objectives, num_individuals, reference_point):
    all_contributions = []
    front_indices = fast_non_dominated_sort(objectives)
    
    for front in front_indices:
        if not front:
            continue
        # 获取前沿内的目标值和种群
        front_objectives = [objectives[idx] for idx in front]
        hv = pg.hypervolume(front_objectives)
        contributions = hv.contributions(reference_point)
        for idx, contrib in zip(front, contributions):
            all_contributions.append((population[idx], objectives[idx], contrib))
    
    # 根据超体积贡献进行排序
    all_contributions.sort(key=lambda x: x[2], reverse=True)

    # 选择前num_individuals个最优个体
    new_population_data = [ind for ind, _, _ in all_contributions[:num_individuals]]
    new_objectives = [obj for _, obj, _ in all_contributions[:num_individuals]]
    
    # 在合并后的种群上进行局部搜索
    new_population_data = perform_local_search(new_population_data)
    
    return new_population_data, new_objectives

def save_data_to_csv(hv_values, throughput_values, delay_values, security_values, filename="simulation_results_lsfv.csv"):
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
    directory = "results-new-ref\\100"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    
    # 保存到CSV
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def main():
    # 设置参数
    pop_size = 100  # 种群大小
    max_gen = 521
    cross_rate = 0.1
    mutation_rate = 0.2
    population = initialize_population(pop_size, N, S)
    # 首次计算适应值，将种群合并为（个体，目标函数值）的形式
    objective_values = [evaluate(ind) for ind in population]
    hv_values = []  # 用于存储每代的HV值
    throughput_values = []
    delay_values = []
    security_values = []
    for gen in range(max_gen):
        if gen % 100 == 0:
            print("gen=", gen)
        # 生成后代
        new_population = []
        for _ in range(25): 
            parent1, parent2 = select_parents(population)
            child1, child2 = sbx_crossover(parent1, parent2, cross_rate)
            child1 = polynomial_mutation(child1, mutation_rate)
            child2 = polynomial_mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
            
        # 交叉变异后的新种群(50个) (individual, objective_value)
        objective_values1 = [evaluate(ind) for ind in new_population]

        # 合并新后代和旧种群
        population = new_population + population
        objective_values += objective_values1
        reference_point = np.max(objective_values, axis=0) + 0.0001         
        hv = pg.hypervolume(objective_values)
        current_hv = hv.compute(reference_point)
        hv_values.append(current_hv)  
                      
        current_throughput = np.mean([ind for ind,_,_ in objective_values])
        throughput_values.append(current_throughput)
        current_delay = np.mean([ind for _,ind,_ in objective_values])
        delay_values.append(current_delay)
        current_security = np.mean([ind for _,_ ,ind in objective_values])
        security_values.append(current_security)
        if gen == max_gen-1:
            avg_throughput = np.mean(current_throughput)
            std_throughput = np.std(current_throughput)
            avg_delay = np.mean(current_delay)
            std_delay = np.std(current_delay)
            avg_security = np.mean(current_security)
            std_security = np.std(current_security)

            print(f"Average Throughput: {avg_throughput}, Std Dev: {std_throughput}")
            print(f"Average Delay: {avg_delay}, Std Dev: {std_delay}")
            print(f"Average Security: {avg_security}, Std Dev: {std_security}")
        
        population, objective_values = update_population_with_hv(population, objective_values, pop_size, reference_point)
    save_data_to_csv(hv_values, throughput_values, delay_values, security_values)
    # print("max hv:", np.max(hv_values))
    # print("min hv:", np.min(hv_values))
    # print("avg hv:", np.average(hv_values))
    # print("time:", time.time()-start_time)
    # save_data_to_csv(hv_values, throughput_values, delay_values, security_values)

    # 绘制HV变化图
    # plt.figure(figsize=(10, 5))
    # plt.plot(hv_values, marker='o', linestyle='-', color='b')
    # plt.title('Hypervolume Over Generations(my)')
    # plt.xlabel('Generation')
    # plt.ylabel('Hypervolume')
    # plt.grid(True)
    # plt.show()

    
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