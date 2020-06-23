import numpy as np


class Schedule:
    def __init__(self, course_id, class_id, teacher_id):
        self.course_id = course_id
        self.class_id = class_id
        self.teacher_id = teacher_id

        self.room_id = 0
        self.week_day = 0
        self.slot = 0

    def random_init(self, room_range):
        self.room_id = np.random.randint(1, room_range + 1, 1)[0]
        self.week_day = np.random.randint(1, 6, 1)[0]
        self.slot = np.random.randint(1, 6, 1)[0]


def schedule_cost(population, elite):
    conflicts = []
    n = len(population[0])

    # 一个pop就是一个课表
    for p in population:
        conflict = 0
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                # 同一个教室在同一个时间只能有一门课，
                if p[i].room_id == p[j].room_id and p[i].week_day == p[j].week_day and p[i].slot == p[j].slot:
                    conflict += 1

                # 同一个班级在同一个时间只能有一门课
                if p[i].class_id == p[j].class_id and p[i].week_day == p[j].week_day and p[i].slot == p[j].slot:
                    conflict += 1

                # 同一个教师在同一个时间只能有一门课
                if p[i].teacher_id == p[j].teacher_id and p[i].week_day == p[j].week_day and p[i].slot == p[j].slot:
                    conflict += 1

                # 同一个班级在同一天不能有相同的课
                if p[i].class_id == p[j].class_id and p[i].course_id == p[j].course_id and p[i].week_day == p[j].week_day:
                    conflict += 1

        conflicts.append(conflict)

    index = np.array(conflicts).argsort()
    return index[: elite], conflicts[index[0]]


