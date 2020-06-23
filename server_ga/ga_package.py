"""
0-1背包问题
一个背包，一组物品，每种物品都有重量和价格，在总重量内，如何选择物品是总价格最高
物体总数 N
背包容量 W
第i个物体重量 w[i]
第i个物体价格 v[i]
"""
import numpy as np
import matplotlib.pyplot as plt
import heapq

def encode(N, unit):
    """
    构建染色体，假设有4个物体，那么染色体长度就是4，01分别表示第i个物体取还是不取
    :param chrom_length:
    :param unit:
    :return:
    """
    unit = int(unit)
    # 0b11001,左边两位取0
    unit_str = str(bin(unit))[2:].zfill(N)

    unit_list = []
    for s in unit_str:
        unit_list.append(s)

    return unit_list

def decode(unit_list):
    """
    二进制转十进制
    :param unit_list:
    :return:
    """
    power = len(unit_list) - 1
    decimal = 0

    for u in unit_list:
        decimal += int(u) * (2 ** power)
        power -= 1

    return decimal

def initPopulation(popu_size, N):
    population = []
    for i in range(popu_size):
        unit_code = encode(N, i)
        population.append(unit_code)
    return np.asarray(population)

def getFitness(N, W, population):
    v_list = []
    for pop in population:
        unit_code = pop
        unit_w = 0
        unit_v = 0
        for j in range(N):
            unit_w += int(unit_code[j]) * w[j]
            unit_v += int(unit_code[j]) * v[j]
        if unit_w <= W:
            v_list.append(unit_v)
        else:
            v_list.append(0)
    return np.array(v_list)


def getElitePopulation(population, fitness_ary, popu_size):
    fitness_list = fitness_ary.tolist()
    elite_index = map(fitness_list.index, heapq.nlargest(popu_size, fitness_list))

    elite_population = np.zeros((popu_size, population.shape[1]))

    i = 0
    for idx in elite_index:
        elite_population[i] = population[idx]
        i += 1
    return elite_population


def crossover(population, cross_prob):
    m, n = population.shape
    cross_num = int(m*cross_prob)
    cross_population = np.zeros((m, n), dtype=np.int)

    if cross_num % 2 != 0:
        cross_num += 1

    cross_index = np.random.choice(range(m), cross_num)


    # 不需要交叉的，直接复制
    for i in range(m):
        if i not in cross_index:
            cross_population[i] = population[i]

    j = 0
    while j < cross_num:
        cross_point = np.random.randint(0, n, 1)[0]

        # 两个染色体后半部分进行交叉
        cross_population[cross_index[j]][0:cross_point] = population[cross_index[j]][0:cross_point]
        cross_population[cross_index[j]][cross_point:] = population[cross_index[j + 1]][cross_point:]

        cross_population[cross_index[j + 1]][0:cross_point] = population[cross_index[j + 1]][0:cross_point]
        cross_population[cross_index[j + 1]][cross_point:] = population[cross_index[j]][cross_point:]
        j += 2

    return cross_population

def mutate(population, mut_prob):
    mut_population = np.copy(population)  # 深拷贝
    m, n = population.shape

    # 需要变异的基因数以及位置
    mut_num = int(m * n * mut_prob)
    mut_index = np.random.choice(range(m * n), mut_num)

    for idx in mut_index:
        row = int(np.floor(idx / n))
        col = idx % n

        if mut_population[row][col] == 0:
            mut_population[row][col] = 1
        else:
            mut_population[row][col] = 0

    return mut_population

def select(population, fitness_ary):
    m, n = population.shape
    new_population = np.zeros((m, n))

    prob_ary = fitness_ary / np.sum(fitness_ary)
    cum_prob_ary = np.cumsum(prob_ary)

    for i in range(m):
        rand = np.random.random()
        # 轮盘选择: 以小于来判定落在了哪一个范围，然后选取
        for j in range(m):
            if rand < cum_prob_ary[j]:
                new_population[i] = population[j]
                break
    return new_population


if __name__ == '__main__':
    max_iter = 50
    N = 4
    # 1个染色体就是1个数，是一个4位的二进制数，表示4个物体各自取的情况
    # 这里2^n相当于把4位二进制的数全部遍历了
    popu_size = pow(2, N)
    w = [2, 3, 1, 5]
    v = [4, 3, 2, 1]
    W = 6

    population = initPopulation(popu_size, N)  # 初始化是没有编码的

    opt_val_list = []
    opt_var_list = []
    for i in range(max_iter):
        print('1')
        fitness_ary = getFitness(N, W, population)

        print('2')
        population = select(population, fitness_ary)

        print('3')
        cross_population = crossover(population, 0.7)

        print('4')
        mut_population = mutate(cross_population, 0.05)

        print('5')
        total_population = np.vstack((population, mut_population))

        print('6')
        total_fitness_ary = getFitness(N, W, total_population)

        print('7')
        population = getElitePopulation(total_population, total_fitness_ary, popu_size)

        best_fitness = max(total_fitness_ary)
        best_index = np.where(total_fitness_ary == best_fitness)[0][0]
        best_pop = total_population[best_index]

        opt_val_list.append(best_fitness)
        opt_var_list.append(best_pop)

        print(f'iter:{i}, fitness:{best_fitness}')

    x = [i for i in range(max_iter)]
    y = [opt_val_list[i] for i in range(max_iter)]
    plt.plot(x, y)
    plt.show()


