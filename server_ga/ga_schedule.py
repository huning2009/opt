import numpy as np
import copy
from Schedule import schedule_cost, Schedule
import prettytable

class GeneOpt:
    def __init__(self, popu_size=30, mut_prob=0.3, elite=5, max_iter=100):
        self.popu_size = popu_size
        self.mut_prob = mut_prob
        self.elite = elite
        self.max_iter = max_iter
        self.population = []

    def init_population(self, data_schedules, room_range):
        for i in range(self.popu_size):
            # 一个entity就表示一个课表，所有班级的排课信息
            entity = []
            # 一个s表示一个班级的上课信息
            for s in data_schedules:
                s.random_init(room_range)
                entity.append(copy.deepcopy(s))

            self.population.append(entity)

    def addSub(self, value, op, value_range):
        """
        对应属性值进行加1和减1操作
        :param value:
        :param op:
        :param value_range:
        :return:
        """
        if op > 0.5:
            if value < value_range:
                value += 1
            else:
                value -= 1
        else:
            if value - 1 > 0:
                value -= 1
            else:
                value += 1

        return value


    def mutate(self, elite_population, room_range):
        e = np.random.randint(0, self.elite, 1)[0]
        ep = copy.deepcopy(elite_population[e])

        for s in ep:
            pos = np.random.randint(0, 3, 1)[0]  # room，week，slot变异选择
            rand = np.random.rand()

            if pos == 0:
                s.room_id = self.addSub(s.room_id, rand, room_range)  # 变异操作

            if pos == 1:
                s.week_day = self.addSub(s.week_day, rand, 5)

            if pos == 2:
                s.slot = self.addSub(s.slot, rand, 5)

        # ep在循环中已经改变了
        return ep

    def crossover(self, elite_population):
        e1 = np.random.randint(0, self.elite, 1)[0]
        e2 = np.random.randint(0, self.elite, 1)[0]

        pos = np.random.randint(0, 2, 1)[0]

        ep1 = copy.deepcopy(elite_population[e1])
        ep2 = elite_population[e2]

        # 这里没交叉，只是单纯的复制，交叉一个
        for p1, p2 in zip(ep1, ep2):
            if pos == 0:
                p1.week_day = p2.week_day
                p1.slot = p2.slot
            if pos == 1:
                p1.room_id = p2.room_id

        return ep1

    def evolution(self, data_schedules, room_range):
        """evolution

        Arguments:
            schedules: class schedules for optimization.
            elite: int, number of best result.

        Returns:
            index of best result.
            best conflict score.
        """
        # Main loop .
        bestScore = 0
        bestSchedule = None

        self.init_population(data_schedules, room_range)

        for i in range(self.max_iter):
            eliteIndex, bestScore = schedule_cost(self.population, self.elite)

            print('Iter: {} | conflict: {}'.format(i + 1, bestScore))

            if bestScore == 0:
                bestSchedule = self.population[eliteIndex[0]]
                break

            # Start with the pure winners
            new_population = [self.population[index] for index in eliteIndex]

            # Add mutated and bred forms of the winners
            while len(new_population) < self.popu_size:
                if np.random.rand() < self.mut_prob:
                    # Mutation
                    newp = self.mutate(new_population, room_range)
                else:
                    # Crossover
                    newp = self.crossover(new_population)

                new_population.append(newp)

            self.population = new_population

        return bestSchedule



def vis(schedule):
    """visualization Class Schedule.

    Arguments:
        schedule: List, Class Schedule
    """
    col_labels = ['week/slot', '1', '2', '3', '4', '5']
    table_vals = [[i + 1, '', '', '', '', ''] for i in range(5)]

    table = prettytable.PrettyTable(col_labels, hrules=prettytable.ALL)

    for s in schedule:
        weekDay = s.week_day
        slot = s.slot
        text = 'course: {} \n class: {} \n room: {} \n teacher: {}'.format(s.course_id, s.class_id, s.room_id, s.teacher_id)
        table_vals[weekDay - 1][slot] = text

    for row in table_vals:
        table.add_row(row)

    print(table)


if __name__ == '__main__':
    schedules = []

    # add schedule
    schedules.append(Schedule(201, 1201, 11101))
    schedules.append(Schedule(201, 1201, 11101))
    schedules.append(Schedule(202, 1201, 11102))
    schedules.append(Schedule(202, 1201, 11102))
    schedules.append(Schedule(203, 1201, 11103))
    schedules.append(Schedule(203, 1201, 11103))
    schedules.append(Schedule(206, 1201, 11106))
    schedules.append(Schedule(206, 1201, 11106))

    schedules.append(Schedule(202, 1202, 11102))
    schedules.append(Schedule(202, 1202, 11102))
    schedules.append(Schedule(204, 1202, 11104))
    schedules.append(Schedule(204, 1202, 11104))
    schedules.append(Schedule(206, 1202, 11106))
    schedules.append(Schedule(206, 1202, 11106))

    schedules.append(Schedule(203, 1203, 11103))
    schedules.append(Schedule(203, 1203, 11103))
    schedules.append(Schedule(204, 1203, 11104))
    schedules.append(Schedule(204, 1203, 11104))
    schedules.append(Schedule(205, 1203, 11105))
    schedules.append(Schedule(205, 1203, 11105))
    schedules.append(Schedule(206, 1203, 11106))
    schedules.append(Schedule(206, 1203, 11106))

    ga = GeneOpt(popu_size=50, elite=10, max_iter=500)
    res = ga.evolution(schedules, 3)

    vis_res = []
    for r in res:
        if r.class_id == 1203:
            vis_res.append(r)

    vis(vis_res)