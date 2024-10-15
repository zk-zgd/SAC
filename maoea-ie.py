import numpy as np
from deap import base, creator, tools
import random
import networkx as nx
from deap.tools.emo import assignCrowdingDist

creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

def calculate_norm_params(population, toolbox):
    # 使用初始评估获取各目标的平均值和标准差
    values = []
    for ind in population:
        # 使用未标准化的评估函数
        throughput_val = throughput(ind)
        delay_val = delay(ind)
        security_val = shard_security(ind)
        values.append((throughput_val, delay_val, security_val))
    values = np.array(values)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    return means, stds

def normalize_fitness(values, means, stds):
    # 应用标准化
    normalized_values = (values - means) / (stds + 1e-15)  # 避免除以零
    return normalized_values

def evaluate(individual, means, stds):
    throughput_val = throughput(individual)
    delay_val = delay(individual)
    security_val = shard_security(individual)
    # 返回未合并的原始目标值
    return (throughput_val, delay_val, security_val)

def initialize_population(pop_size, num_nodes, num_shards):
    population = []
    for _ in range(pop_size):
        # 确保每个分片至少有4个节点
        while True:
            # 初始化分片计数器
            shard_counts = {i: 0 for i in range(1, num_shards + 1)}
            
            # 生成随机分片配置
            individual_data = np.random.choice(range(1, num_shards + 1), num_nodes, replace=True)
            
            # 计算每个分片的节点数
            for shard in individual_data:
                shard_counts[shard] += 1
            
            # 检查每个分片的节点数是否至少为4
            if all(count >= 12 for count in shard_counts.values()):
                break

        # 创建个体
        individual = creator.Individual(individual_data)
        population.append(individual)
    return population

def custom_mutate(individual, num_shards, min_per_shard=4):
    # 随机选择变异位置
    index_to_mutate = random.randint(0, len(individual) - 1)
    possible_values = list(range(1, num_shards + 1))
    current_value = individual[index_to_mutate]
    possible_values.remove(current_value)
    
    # 临时修改并检查
    old_value = individual[index_to_mutate]
    shard_counts = {i: individual.count(i) for i in range(1, num_shards + 1)}
    # 确保新的值不会违反分片的最小节点限制
    if shard_counts[old_value] > min_per_shard:
        individual[index_to_mutate] = random.choice(possible_values)
        new_shard_counts = {i: individual.count(i) for i in range(1, num_shards + 1)}
        if any(count < min_per_shard for count in new_shard_counts.values()):
            individual[index_to_mutate] = old_value  # 如果变异后有问题，撤销变异

    return individual,

toolbox.register("initialize", initialize_population, pop_size=100, num_nodes=100, num_shards=5)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", custom_mutate)

def calculate_entropy(population):
    # 提取种群中所有个体的适应度值
    fitness_values = np.array([ind.fitness.values for ind in population])
    # 正规化适应度值以防止计算问题
    normalized_fitness = (fitness_values - fitness_values.min(axis=0)) / (fitness_values.max(axis=0) - fitness_values.min(axis=0) + 1e-15)
    # 计算每个适应度的概率分布
    sums = normalized_fitness.sum(axis=0)
    # 为避免除以零，给分母添加一个小的常数
    p = normalized_fitness / (sums + 1e-15)
    # 计算信息熵
    entropy = -np.sum(p * np.log2(p + 1e-15), axis=0)
    return entropy.sum()  # 返回总熵值，汇总所有目标的熵

def select_based_on_entropy(population, size, entropy_threshold=0.5):
    current_entropy = calculate_entropy(population)
    if (current_entropy < entropy_threshold).all():
        # 如果熵低于阈值，增加竞争者的数量，以增加选择压力
        return tools.selTournament(population, size, tournsize=int(len(population) / 5))
    else:
        # 如果熵高于或等于阈值，使用标准锦标赛选择
        return tools.selTournament(population, size, tournsize=3)

def evolve_population(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    means, stds = calculate_norm_params(population, toolbox)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    entropy_threshold = 0.5  # 设定信息熵阈值

    for gen in range(1, ngen + 1):
        # 评估适应度
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind, means, stds)  # 使用标准化评估

        # 基于信息熵选择下一代个体
        offspring = select_based_on_entropy(population, len(population), entropy_threshold)
        offspring = list(map(toolbox.clone, offspring))

        # 应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant, S)
                del mutant.fitness.values

        # 重新评估新产生的个体适应度
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, means, stds)

        # 更新种群
        population[:] = offspring

        # 计算拥挤距离
        assignCrowdingDist(population)
        halloffame.update(population)

        # 只在最后一代进行记录
        if gen == ngen:
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(offspring), **record)
            if verbose:
                print(logbook.stream)

    return population, logbook

# 定义统计工具
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg_throughput", lambda pop: np.mean([ind[0] for ind in pop]))
stats.register("std_throughput", lambda pop: np.std([ind[0] for ind in pop]))
stats.register("min_throughput", lambda pop: np.min([ind[0] for ind in pop]))
stats.register("max_throughput", lambda pop: np.max([ind[0] for ind in pop]))

stats.register("avg_delay", lambda pop: np.mean([ind[1] for ind in pop]))
stats.register("std_delay", lambda pop: np.std([ind[1] for ind in pop]))
stats.register("min_delay", lambda pop: np.min([ind[1] for ind in pop]))
stats.register("max_delay", lambda pop: np.max([ind[1] for ind in pop]))
    
stats.register("avg_security", lambda pop: np.mean([ind[2] for ind in pop]))
stats.register("std_security", lambda pop: np.std([ind[2] for ind in pop]))
stats.register("min_security", lambda pop: np.min([ind[2] for ind in pop]))
stats.register("max_security", lambda pop: np.max([ind[2] for ind in pop]))

# 创建 Logbook
logbook = tools.Logbook()
logbook.header = ['gen'] + stats.fields

def main():
    population = toolbox.initialize()
    cxpb, mutpb, ngen = 0.9, 0.1, 521
    halloffame = tools.HallOfFame(10)
    
    final_population, logbook = evolve_population(population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose=True)

    # 输出最优个体
    top_individuals = halloffame.items
    for ind in top_individuals:
        print("Fitness:", ind.fitness.values)
        print("ind:", ind)

        index_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
        # Populate the dictionary with indices
        for index, value in enumerate(ind):
            if value in index_dict:
                index_dict[value].append(index)

        # Print the dictionary
        for key, indices in index_dict.items():
            print(f"Key {key}: Indices {indices}")

if __name__ == "__main__":
    main()
