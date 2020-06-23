import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import heapq

'''
求多项式极值

    max(x1,x2) = 21.5+x1*sin(4pi*x1)+x2*sin(20pi*x2)

    s.t. -3.0<=x1<=12.1
          4.1<=x2<=5.8
          
代码中染色体和种群全部用numpy表示，不要使用list
'''


class GaOpt:
    def __init__(self, var_bounds, delta, popu_size, cross_prob, mut_prob, max_iter):
        """
        var_bound: 变量范围，数组，每个元素即一个变量的上下界
        """
        self.var_bounds = var_bounds  # 变量上下界
        self.delta = delta  # 精度
        self.popu_size = popu_size  # 种群数量
        self.cross_prob = cross_prob  # 新种群数量
        self.mut_prob = mut_prob  # 变异率
        self.max_iter = max_iter  # 最大迭代次数
        self.opt_val_list = []
        self.opt_var_list = []

    def getEncodeLengths(self):
        """
        编码长度，定义种群、解码计算需要
        :return:
        """
        lengths = []
        for bound in self.var_bounds:
            uper = bound[1]
            low = bound[0]
            # 求解变量编码长度：
            # 区间[1,3]，精度0.01, 至少需要(3-1)/0.01=200长度来表示
            # 200长度对于二进度 2^x=200 -> x=8
            res = fsolve(lambda x: ((uper - low) / self.delta - 2 ** x + 1), 30)
            var_len = int(np.ceil(res[0]))
            lengths.append(var_len)
        return lengths

    def getDecodePopulation(self, population, encode_lengths):
        """
        把二进制转为十进制
        :param population: 二进制种群
        :param encode_lengths: 每个变量编码后的长度，[18,15]
        :param var_bounds: 每个变量的上下界
        :return:
        """
        popu_size = population.shape[0]  # 种群个数
        var_num = len(encode_lengths)  # 变量个数

        # 种群解码
        decode = np.zeros((popu_size, var_num), dtype=np.float)

        for i, pop in enumerate(population):
            start = 0
            # 对每一个变量进行解码
            for j, length in enumerate(encode_lengths):
                '''
                比如2个变量长度分别是[18, 15]
                转为二进制次方就是 2^17,2^16,...1， 
                start是标记染色体基因的位置，
                '''
                power = length - 1
                decimal = 0

                for k in range(start, start + length):
                    decimal += (pop[k] * (2 ** power))
                    power -= 1

                start = length

                uper = self.var_bounds[j][1]
                low = self.var_bounds[j][0]

                # 把解码的值归一化到指定范围内
                val = low + decimal * (uper - low) / (2 ** length - 1)
                decode[i][j] = val
        return decode

    def initPopulation(self, encode_lengths):
        """
        根据变量上下界确定染色体长度，构建种群
        encode_lengths: 每个变量的编码长度
        :return:
        """
        chrom_len = sum(encode_lengths)
        population = np.zeros((self.popu_size, chrom_len), dtype=np.int)
        for i in range(self.popu_size):
            # 随机生成二进制01.
            population[i, :] = np.random.randint(0, 2, chrom_len)
        return population

    def getFitness(self, decode_population):
        """
        适应度函数，十进制种群，再按目标函数计算适应度
        :return:
        """
        # max(21.5 + x1*sin(4pi*x1) + x2*sin(20pi*x2)
        fitness_fn = lambda x1, x2: 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)

        pop_size = len(decode_population)
        fitness_list = []
        for i in range(pop_size):
            tmp_x1 = decode_population[i][0]
            tmp_x2 = decode_population[i][1]
            fitness = fitness_fn(tmp_x1, tmp_x2)
            fitness_list.append(fitness)

        fitness_ary = np.asarray(fitness_list)
        return fitness_ary

    def select(self, population, fitness_ary):
        """
        根据轮盘选择最优种群：
        id：    1，   2，    3，    4，    5
        prob:  0.4， 0.2，  0.1，  0.1，  0，2
        cum：  0.4   0.6    0.7   0.8    1.0

        丢随机数，0-0.4的选1，0.4-0.6的选2，以此类推，概率大的，cum的长度就大，
        :return:
        """

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

    def crossover(self, new_population):
        """
        交叉：是染色体之间进行交叉
        :return:
        """
        m, n = new_population.shape
        cross_num = int(m * self.cross_prob)

        if cross_num % 2 != 0:
            cross_num += 1

        cross_population = np.zeros((m, n), dtype=np.int)

        cross_index = np.random.choice(range(m), cross_num)

        # 不需要交叉的，直接复制
        for i in range(m):
            if i not in cross_index:
                cross_population[i] = new_population[i]

        j = 0
        while j < cross_num:
            cross_point = np.random.randint(0, n, 1)[0]

            # 两个染色体后半部分进行交叉
            cross_population[cross_index[j]][0:cross_point] = new_population[cross_index[j]][0:cross_point]
            cross_population[cross_index[j]][cross_point:] = new_population[cross_index[j+1]][cross_point:]

            cross_population[cross_index[j+1]][0:cross_point] = new_population[cross_index[j+1]][0:cross_point]
            cross_population[cross_index[j+1]][cross_point:] = new_population[cross_index[j]][cross_point:]

            j += 2

        return cross_population

    def mutate(self, cross_population):
        """
        变异: 是整个种群的随机基因数变异
        :return:
        """
        mut_population = np.copy(cross_population)  # 深拷贝
        m, n = cross_population.shape

        # 需要变异的基因数以及位置
        mut_num = int(m * n * self.mut_prob)
        mut_index = np.random.choice(range(m*n), mut_num)

        for idx in mut_index:
            row = int(np.floor(idx / n))
            col = idx % n

            if mut_population[row][col] == 0:
                mut_population[row][col] = 1
            else:
                mut_population[row][col] = 0

        return mut_population


    def getElitePopulation(self, decode_population, fitness_ary):
        """
        找到最好的popu_size个染色形成新的种群
        :param population:
        :param fitness_ary:
        :return:
        """
        # 转为list，要和下标结合使用
        fitness_list = fitness_ary.tolist()
        elite_index = map(fitness_list.index, heapq.nlargest(self.popu_size, fitness_list))

        elite_population = np.zeros((self.popu_size, decode_population.shape[1]))

        i = 0
        for idx in elite_index:
            elite_population[i] = decode_population[idx]
            i += 1
        return elite_population

    def evaluate(self):
        opt_val_list = []
        opt_var_list = []
        encode_lengths = self.getEncodeLengths()
        population = self.initPopulation(encode_lengths)
        for i in range(self.max_iter):

            decode_population = self.getDecodePopulation(population, encode_lengths)
            fitness_ary = self.getFitness(decode_population)
            new_population = self.select(population, fitness_ary)
            cross_population = self.crossover(new_population)
            mut_population = self.mutate(cross_population)

            total_population = np.vstack((population, mut_population))

            total_decode_population = self.getDecodePopulation(total_population, encode_lengths)
            total_fitness_ary = self.getFitness(total_decode_population)

            population = self.getElitePopulation(total_population, total_fitness_ary)

            # 最优的适应度
            opt_val_list.append(np.max(total_fitness_ary))
            index = np.where(opt_val_list == max(total_fitness_ary))[0][0]
            # 最优的适应度对应的解
            opt_var_list.append(list(total_decode_population[index]))

            print(f'iter:{i}, best fiteness:{max(total_fitness_ary)}')

        self.opt_val_list = opt_val_list
        self.opt_var_list = opt_var_list
        print('iter over!')

    def print(self):
        x = [i for i in range(self.max_iter)]
        y = [self.opt_val_list[i] for i in range(self.max_iter)]
        plt.plot(x, y)
        plt.show()

    def get_best_val(self):
        best_val = np.max(self.opt_val_list)
        index = np.where(best_val == max(self.opt_val_list))[0][0]
        best_var = self.opt_var_list[index]
        return best_val, best_var


if __name__ == '__main__':
    var_bounds = [[-3.0, 12.1], [4.1, 5.8]]
    delta = 0.0001
    popu_size = 100
    cross_prob = 0.8
    mut_prob = 0.01
    max_iter = 100
    ga_opt = GaOpt(var_bounds, delta, popu_size, cross_prob, mut_prob, max_iter)
    ga_opt.evaluate()
    ga_opt.print()