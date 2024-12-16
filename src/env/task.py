import numpy as np
from src.utils.last_finsih_time import calculate_latest_finish_time

from src.config.env_config import task_config
class Subtask:
    def __init__(self, task, subtask_id, subtask_location, subtask_skill_list, subtask_spend_time):
        self.task = task
        self.id = subtask_id
        self.location = subtask_location
        self.skill_list = subtask_skill_list
        self.time = subtask_spend_time

        self.is_finished = False

        self.is_assigned = False
        self.assigned_worker = None

        self.time_to_complete = 0

        self.is_waiting = True

        self.waiting_time = 0

        self.last_finish_time = 0
        self.last_finish_expected_time = 0

        self.ind_reward = 0
        self.group_reward = 0

        self.max_reward = 0
        self.min_reward = 0

        self.u1 = 0
        self.u2 = 0
        self.u3 = 0

    def get_graph_matrix(self):

        un_do_subtask_list = [self.task.subtask_list[i].id for i in self.task.un_subtask_list]

        node_order = un_do_subtask_list.index(self.id)

        un_do_inf = []
        for i in un_do_subtask_list:
            wait = 1 if self.task.subtask_list[i].is_waiting else 0
            un_do_sub = [wait, self.task.subtask_list[i].time, self.task.subtask_list[i].last_finish_expected_time, self.task.subtask_list[i].last_finish_time, self.task.subtask_list[i].location, self.task.subtask_list[i].skill_list]
            un_do_sub = flatten(un_do_sub)

            un_do_inf.append(un_do_sub)

        size = len(self.task.un_subtask_list)
        new_graph_matrix = [[0] * size for _ in range(size)]

        for i in range(size):
            for j in range(size):
                if self.task.graph_matrix[self.task.un_subtask_list[i]][self.task.un_subtask_list[j]] == 1:
                    new_graph_matrix[i][j] = 1

        return un_do_inf, new_graph_matrix, node_order

    def get_inf_self(self):
        wait = 1 if self.is_waiting else 0
        return flatten(
            [wait, self.time/200, self.last_finish_expected_time/5000, self.last_finish_time/5000, self.location[0]-30.2,self.location[1]-103.6, self.skill_list])


"""-----------------Task-----------------"""


class Task:
    def __init__(self, task_id, subtask_information, subtask_graph_matrix, task_deadline,
                 task_expected_time, task_commission,time_discount):
        self.id = task_id
        self.graph_matrix = subtask_graph_matrix

        self.deadline = task_deadline
        self.expected_time = task_expected_time
        self.commission = task_commission

        self.time_discount = time_discount

        self.time = 0


        (self.subtask_list,
         self.un_assigned_subtask_num) = self.initialize_subtask(subtask_information)

        self.available_subtask_list = []

        self.no_pre_subtask_list = []

        self.no_pre_matrix = self.graph_matrix.copy()

        self.initialize_subtask_list()

        self.finished_subtask_list = []
        self.is_assigned_subtask_list = []
        self.un_subtask_list = list(range(self.un_assigned_subtask_num))

        self.finish = False

        self.movement_discount = 0
        self.real_commission = 0

    def initialize_subtask(self, subtask_information):
        # 初始化子任务
        subtask_list = []
        subtask_time_list = []
        id_seq = 0

        subtask_num = len(subtask_information)

        min_reward = (self.commission - self.time_discount * (self.deadline - self.expected_time)) / subtask_num

        max_reward = self.commission / subtask_num

        for subtask in subtask_information:
            # todo
            sub_task = Subtask(self, id_seq, subtask[0], subtask[1],
                               subtask[2])
            id_seq += 1

            sub_task.max_reward = max_reward
            sub_task.min_reward = min_reward

            subtask_list.append(sub_task)
            subtask_time_list.append(sub_task.time)

        last_finish_list = calculate_latest_finish_time(self.graph_matrix, subtask_time_list, self.deadline)
        last_expected_finish_list = calculate_latest_finish_time(self.graph_matrix, subtask_time_list,
                                                                 self.expected_time)
        for sub_task in subtask_list:
            sub_task.last_finish_time = last_finish_list[sub_task.id]
            sub_task.last_finish_expected_time = last_expected_finish_list[sub_task.id]

        return subtask_list, subtask_num

    def initialize_subtask_list(self):
        # 初始化可用子任务列表
        self.no_pre_subtask_list = []
        self.available_subtask_list =[]
        for i in self.subtask_list:
            if np.all(self.no_pre_matrix[:, i.id] == 0):
                self.no_pre_subtask_list.append(i.id)
                i.is_waiting = False
            if np.all(self.graph_matrix[:, i.id] == 0):
                self.available_subtask_list.append(i.id)


    def subtask_finish_update(self, subtask_id):
        # after subtask finish, update available_subtask_list and graph_matrix

        self.finished_subtask_list.append(subtask_id)
        for i in range(self.no_pre_matrix.shape[0]):
            if self.no_pre_matrix[subtask_id, i] == 1:
                self.no_pre_matrix[subtask_id, i] = 0
                if np.all(self.no_pre_matrix[:, i] == 0):
                    self.no_pre_subtask_list.append(i)
                    self.subtask_list[i].is_waiting = False

    def after_matching_update_sub(self, subtask_id, worker, distance, speed, worker_movement_penalty):
        self.subtask_list[subtask_id].is_assigned = True
        self.subtask_list[subtask_id].assigned_worker = worker

        self.subtask_list[subtask_id].time_to_complete = max(distance / speed,
                                                             self.subtask_list[subtask_id].waiting_time) + \
                                                         self.subtask_list[subtask_id].time

        self.is_assigned_subtask_list.append(subtask_id)
        self.un_assigned_subtask_num -= 1
        self.available_subtask_list.remove(subtask_id)
        self.un_subtask_list.remove(subtask_id)
        if subtask_id in self.no_pre_subtask_list:
            self.no_pre_subtask_list.remove(subtask_id)
        self.movement_discount -= distance * worker_movement_penalty

        # individual reward
        if self.subtask_list[subtask_id].time_to_complete <= self.subtask_list[subtask_id].last_finish_expected_time:
            self.subtask_list[subtask_id].ind_reward = self.subtask_list[
                                                           subtask_id].max_reward - worker_movement_penalty * distance
        elif self.subtask_list[subtask_id].time_to_complete <= self.subtask_list[subtask_id].last_finish_time:
            self.subtask_list[subtask_id].ind_reward = self.subtask_list[
                                                           subtask_id].min_reward - worker_movement_penalty * distance
        else:
            self.subtask_list[subtask_id].ind_reward = - worker_movement_penalty * distance

        for i in range(self.graph_matrix.shape[0]):
            if self.graph_matrix[subtask_id, i] == 1:
                self.graph_matrix[subtask_id, i] = 0
                self.subtask_list[i].waiting_time = max(self.subtask_list[i].waiting_time,
                                                        self.subtask_list[subtask_id].time_to_complete)
                if np.all(self.graph_matrix[:, i] == 0):
                    self.available_subtask_list.append(i)
                    self.subtask_list[i].is_waiting = False

        if self.un_assigned_subtask_num == 0:
            last_finish_time = 0
            for i in self.is_assigned_subtask_list:
                if self.subtask_list[i].time_to_complete > last_finish_time:
                    last_finish_time = self.subtask_list[i].time_to_complete
            self.time += last_finish_time

            self.finish = True

            # group reward
            max_time = 0
            # for i in self.is_assigned_subtask_list:
            #     if self.subtask_list[i].time_to_complete > max_time:
            #         max_time = self.subtask_list[i].time_to_complete

            max_time += self.time

            if max_time <= self.expected_time:
                self.subtask_list[subtask_id].group_reward = self.commission + self.movement_discount
            elif max_time <= self.deadline:
                self.subtask_list[subtask_id].group_reward = self.commission - self.time_discount * (
                        max_time - self.expected_time) + self.movement_discount
            else:
                self.subtask_list[subtask_id].group_reward = self.movement_discount

            self.real_commission = self.subtask_list[subtask_id].group_reward


def flatten(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list) or isinstance(element, tuple):
            flat_list.extend(flatten(element))
        else:
            flat_list.append(element)
    return flat_list
