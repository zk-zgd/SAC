from deap import base, creator, tools, algorithms
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pygmo as pg

# 定义适应度类和个体类
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))  # 优化目标：最大化吞吐量，最小化延迟和安全性风险
creator.create("Individual", list, fitness=creator.FitnessMax)

# 设置工具箱
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 5)  # 分片编号从1到5
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=100)  # 个体由100个节点组成
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册交叉、变异和选择操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.05)  # 个体每个基因的变异概率为5%
toolbox.register("select", tools.selNSGA2)

def get_distances(G):
    # 为每个节点分配随机坐标（例如在[0, 100] x [0, 100]区域内）
    for node in G.nodes():
        G.nodes[node]['pos'] = (np.random.rand() * 100, np.random.rand() * 100)
        G.nodes[node]['reputation'] = initial_reputation
        G.nodes[node]['history'] = []

    # 计算并存储每对节点之间的距离
    distances = {}
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                pos1 = np.array(G.nodes[node1]['pos'])
                pos2 = np.array(G.nodes[node2]['pos'])
                # 计算欧几里得距离
                distance = np.linalg.norm(pos1 - pos2)
                distances[(node1, node2)] = distance
    return distances

# 模拟节点行为和更新信誉值
def simulate_behavior_and_update_reputation(G, epochs=50):
    window_size = 10  # 滑动窗口大小
    alpha = 0.3  # 指数加权的参数
    for _ in range(epochs):
        for node in G.nodes():
            # 模拟节点行为：1表示诚实，0表示不诚实
            action = np.random.choice([1, 0], p=[0.8, 0.2])  # 假设80%概率诚实
            G.nodes[node]['history'].append(action)
            
            # 更新信誉值，使用滑动窗口和指数加权平均
            history = G.nodes[node]['history'][-window_size:]  # 取最近的窗口期行为
            weighted_sum = sum(alpha * (1 - alpha) ** i * x for i, x in enumerate(reversed(history)))
            weighted_average = weighted_sum / sum(alpha * (1 - alpha) ** i for i in range(len(history)))
            G.nodes[node]['reputation'] = weighted_average
    return [G.nodes[node]['reputation'] for node in G.nodes()]

# 根据历史交易记录，计算节点处理效率：节点所在分片每秒可以处理的最大交易数
def get_node_effciency(shard_transactions, shard_time):
    # 计算每个分片的处理效率
    shard_efficiencies = {shard: shard_transactions[shard] / shard_time[shard] for shard in shard_transactions}
    return shard_efficiencies

# 参数
N = 100 # 节点数
p = 0.05  # 连接概率
initial_reputation = 0.5  # 初始信誉值
shard_transactions = {1: 1200, 2: 1500, 3: 1800, 4: 1600, 5: 1400}  # 每个分片处理的交易数
shard_time = {1: 20, 2: 10, 3: 5, 4: 8, 5: 4}  # 对应的时间长度（秒）
node_shard_history = np.random.choice(range(1, 6), 100)
S = 5 # 分片个数
G = nx.erdos_renyi_graph(N, p) # 创建图
distances = get_distances(G) # 网络拓扑结构（节点间的物理距离）
reputations = simulate_behavior_and_update_reputation(G) # 节点的信誉值
security_thres = 2/3 # 安全性阈值
# 根据每个节点原始所在分片的效率，为每个节点分配其历史效率
shard_efficiencies = get_node_effciency(shard_transactions, shard_time)
node_efficiencies = [shard_efficiencies[node_shard_history[i]] for i in range(len(node_shard_history))]

def get_nodes_per_shard(individual, shard_count):
    """
    构建每个分片内的节点列表。
    :param individual: 每个节点的分片分配列表。
    :param shard_count: 分片的总数。
    :return: 每个分片内的节点列表的字典。
    """
    nodes_per_shard = {shard: [] for shard in range(1, shard_count + 1)}
    for node_index, shard_index in enumerate(individual):
        if shard_index in nodes_per_shard:
            nodes_per_shard[shard_index].append(node_index)
        else:
            print("Invalid shard index:", shard_index)
    return nodes_per_shard

def throughput(individual):
    """
    计算总吞吐量。
    :param individual: 每个节点的分片分配。
    :return: 总吞吐量。
    """
    nodes_per_shard = get_nodes_per_shard(individual, S)  # 使用辅助函数构建字典

    total_throughput = 0
    for _, nodes in nodes_per_shard.items():
        if nodes:
            efficiencies = sum(node_efficiencies[node] for node in nodes)
            shard_throughput = efficiencies / len(nodes)  # 计算分片的平均效率
            total_throughput += shard_throughput

    return total_throughput

def delay(individual):
    # 需要先计算节点间实际通讯的平均延迟
    total_delay = 0
    count = 0
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            if individual[i] == individual[j]:  # 只计算同一分片内的节点间延迟
                total_delay += distances[(i, j)]
                count += 1

    average_delay = total_delay / count if count != 0 else float('inf')  # 避免除以0
    return -average_delay

def shard_security(individual):
    """
    计算分片的安全性。
    :param individual: 每个节点的分片分配。
    :return: 分片安全性评分（负值标准差，我们希望最小化标准差）。
    """
    nodes_per_shard = get_nodes_per_shard(individual, S)  # 使用辅助函数构建字典

    malicious_counts = []
    for shard, nodes in nodes_per_shard.items():
        malicious_nodes = sum(1 for node in nodes if reputations[node] < security_thres)
        malicious_counts.append(malicious_nodes)

    malicious_std = np.std(malicious_counts) if malicious_counts else 0

    return -malicious_std

def evaluate(individual):
    throughput_val = throughput(individual)
    delay_val = delay(individual)
    security_val = shard_security(individual)
    return throughput_val, delay_val, security_val

toolbox.register("evaluate", evaluate)

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

def update_population_with_hv(population, objectives, num_individuals, reference_point):
    all_contributions = []
    front_indices = fast_non_dominated_sort(objectives)
    
    for front in front_indices:
        if not front:
            continue
        # 获取前沿内的目标值和种群
        front_objectives = [objectives[idx] for idx in front]
        front_population = [population[idx] for idx in front]
        front_distances = calculate_crowding_distance(front_objectives)

        # 计算每个个体在前沿的超体积贡献
        contributions = calculate_hypervolume_contributions(front_population, front_objectives, reference_point)

        # 添加贡献值和距离
        for local_idx, (contrib, dist) in enumerate(zip(contributions, front_distances)):
            global_idx = front[local_idx]  # 将局部索引映射回全局索引
            all_contributions.append((front_population[local_idx], front_objectives[local_idx], contrib[2], dist))

    # 根据超体积贡献和拥挤距离进行排序
    all_contributions.sort(key=lambda x: (x[2], x[3]), reverse=True)

    # 选择前num_individuals个最优个体
    new_population = [ind for ind, _, _, _ in all_contributions[:num_individuals]]
    new_objectives = [obj for _,obj,_,_ in all_contributions[:num_individuals]]
    
    return new_population, new_objectives

def main():
    population = toolbox.population(n=100)  # 创建种群
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    cxpb, mutpb, ngen = 0.5, 0.2, 521       # 交叉概率、变异概率和代数

    for gen in range(ngen):
        if gen % 200 == 0:
            print("gen=", gen)
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        full_population = offspring + population
        objectives = [ind.fitness.values for ind in full_population]
        reference_point = np.max(objectives, axis=0) + 0.0001
        # Update the population using HV contributions
        population[:], objectives[:] = update_population_with_hv(full_population, objectives, len(population), reference_point)

    return population, objectives


if __name__ == "__main__":
    final_pop, final_objs = main()
    security_threshold = -1
    secure_individuals = [ind for ind in final_pop if ind.fitness.values[2] <= security_threshold]
    # 检查是否有满足条件的个体
    if not secure_individuals:
        print("No individuals meet the security threshold.")
    else:
        # 在满足安全性的个体中选择吞吐量最高和时延最低的个体
        # 这里假设适应度值顺序为：吞吐量（要最大化），时延（要最小化），安全性（已筛选）
        best_individual = max(secure_individuals, key=lambda ind: (ind.fitness.values[0], -ind.fitness.values[1]))
        if best_individual:
            print("Best Individual:", list(best_individual))
            print("Fitness Values:", best_individual.fitness.values)
        
