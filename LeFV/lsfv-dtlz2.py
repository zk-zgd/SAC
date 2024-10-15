import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import time

def g_function(X_m):
    return sum((x - 0.5)**2 for x in X_m)

def objective_1(X, num_objectives=3):
    assert len(X) >= num_objectives, "Length of decision vector should be at least equal to number of objectives"
    g = g_function(X[num_objectives-1:])
    return (1 + g) * np.prod(np.cos(X[:1] * np.pi / 2))

def objective_2(X, num_objectives=3):
    assert len(X) >= num_objectives, "Length of decision vector should be at least equal to number of objectives"
    g = g_function(X[num_objectives-1:])
    return (1 + g) * np.prod(np.cos(X[:1] * np.pi / 2)) * np.sin(X[1] * np.pi / 2)

def objective_3(X, num_objectives=3):
    assert len(X) >= num_objectives, "Length of decision vector should be at least equal to number of objectives"
    g = g_function(X[num_objectives-1:])
    return (1 + g) * np.prod(np.cos(X[:2] * np.pi / 2)) * np.sin(X[2] * np.pi / 2)

def evaluate_population(population, objective_functions):
    objectives = []
    for individual in population:
        individual_objectives = [func(individual) for func in objective_functions]
        objectives.append((individual, individual_objectives))
    return objectives

def initialize_population(pop_size, num_nodes):
    population = []
    for _ in range(pop_size):
        individual = np.random.uniform(0, 1, num_nodes)
        population.append(individual)
    return population

# 分布指数eta_c越大，子代越趋向于父代
def sbx_crossover(parent1, parent2, prob_crossover=0.05, eta_c=15):
    child1, child2 = parent1.copy(), parent2.copy()
    
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


def polynomial_mutation(individual, prob_mutation=0.01, eta_m=20):
    """多项式变异"""    
    for i in range(len(individual)):
        if np.random.rand() < prob_mutation:
            u = np.random.rand()
            delta = np.where(u < 0.5, (2*u)**(1/(eta_m+1)) - 1, 1 - (2*(1-u))**(1/(eta_m+1)))
            individual[i] += delta
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

def calculate_hypervolume_contributions(front, reference_point):
    """计算前沿层中每个个体的HV贡献"""
    hv = pg.hypervolume(front)
    contributions = hv.contributions(reference_point)
    return contributions

def calculate_reference_point(objectives):
    num_objectives = len(objectives[0])  
    reference_point = []
    for i in range(num_objectives):
        if i == 0:
            max_value = max(ind[i] for ind in objectives)
            reference_point.append(max_value + 100)
        elif i == 1:
            min_value = max(ind[i] for ind in objectives)
            reference_point.append(min_value + 10)
        elif i == 2:
            min_value = max(ind[i] for ind in objectives)
            reference_point.append(min_value + 6)
    return reference_point


def update_population_with_hv(population, objectives, num_individuals, reference_point):
    all_contributions = []
    # 将150个目标函数值按快速非支配排序分开计算contris
    for front in fast_non_dominated_sort(objectives):
        if not front:
            continue
        front_values = [objectives[idx] for idx in front]
        contributions = calculate_hypervolume_contributions(front_values, reference_point)
        for idx, contrib in zip(front, contributions):
            all_contributions.append((population[idx], objectives[idx], contrib))
    
    # Sort by hypervolume contributions
    all_contributions.sort(key=lambda x: x[2], reverse=True)
    
    # 取100行之前的，保留为新种群((个体，目标值)，目标值，贡献)
    new_population = all_contributions[:num_individuals]
    # (个体, 目标值)
    new_population_data = [ind for ind, obj, _ in all_contributions[:num_individuals]]
    
    # 取100行之后的50个个体((个体，目标值)，目标值，贡献)
    remained_individuals = all_contributions[num_individuals:]
    # 进行局部搜索优化以去除重复的个体 (个体，目标值)
    remained_population = perform_local_search(remained_individuals)

    # 截取除了最后len(remained_population)个元素外的部分
    new_population_data = new_population_data[:-len(remained_population)]
    # 将remained_population追加到new_population_data后面
    new_population_data.extend(remained_population)
    
   
    new_population = new_population_data
    new_objectives = [obj for _, obj in new_population_data]

    return new_population, new_objectives


def perform_local_search(remained_individuals):
    unique_individuals = {}
    for individual, objectives, _ in remained_individuals:
        pop = individual[0]
        if isinstance(pop, np.ndarray):
            ind_tuple = tuple(pop)
        else:
            continue
        key = (ind_tuple, tuple(objectives))
        if key not in unique_individuals:
            unique_individuals[key] = (individual, objectives)
    ret = []
    for key, (ind, obj) in unique_individuals.items():
        # 将 ind_tuple 转换回 numpy array
        array_ind = np.array(ind[0])
        ret.append((array_ind, obj))
    return ret

def main(start_time):
    # 设置参数
    # num_objectives = 3  # 目标函数的数量
    pop_size = 100  # 种群大小
    # num_generations = 921
    N = 100
    num_generations = 280

    # 初始化种群
    raw_population = initialize_population(pop_size, N)
    funcs = [objective_1, objective_2, objective_3]
    # 首次计算适应值，将种群合并为（个体，目标函数值）的形式
    population = evaluate_population(raw_population, funcs)
    hv_values = []  # 用于存储每代的HV值
    for gen in range(num_generations):
        # 获取种群的目标值列表
        objective_values = []
        for entry in population:
            if len(entry) >= 2:  # 确保至少有两个元素
                objectives = entry[1]  # 假定目标值总是在第二位置
                objective_values.append(objectives)
            else:
                print("Unexpected structure:", entry)  # 打印结构不符的元素
        # 计算种群的参考点
        reference_point = calculate_reference_point(objective_values)
        # 除了第一次外，基于种群中每个个体的contributions更新种群
        if len(objective_values) > 100:
            # population = (individual, objective_value)
            population, objective_values = update_population_with_hv(population, objective_values, pop_size, reference_point)
        
            # 更新参考点
            reference_point = calculate_reference_point(objective_values)

        # 计算当前种群的HV
        hv = pg.hypervolume(objective_values)
        current_hv = hv.compute(reference_point)
        hv_values.append(current_hv)
        new_raw_population = []
        while len(new_raw_population) < 1/2 * len(population):
            parent1, parent2 = select_parents(population)
            if len(parent1) == 100 and len(parent2) == 1:
                child1, child2 = sbx_crossover(parent1, parent2)
            elif len(parent1) == 2 and len(parent2) == 2:
                child1, child2 = sbx_crossover(parent1[0], parent2[0])
            else:
                print("len(parent1)", len(parent1))
                print("len(parent2)", len(parent2))
            child1 = polynomial_mutation(child1)
            child2 = polynomial_mutation(child2)
            new_raw_population.extend([child1, child2])
        
        # 交叉变异后的新种群(50个) (individual, objective_value)
        new_population = evaluate_population(new_raw_population, funcs)
        if len(population[0]) == 100:
            population = evaluate_population(population, funcs)

        # 合并新后代和旧种群
        population = new_population + population

    print("max hv:", np.max(hv_values))
    print("min hv:", np.min(hv_values))
    print("avg hv:", np.average(hv_values))
    print("time:", time.time()-start_time)
    # 绘制HV变化图
    plt.figure(figsize=(10, 5))
    plt.plot(hv_values, marker='o', linestyle='-', color='b')
    plt.title('Hypervolume Over Generations(my)')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    main(start_time)