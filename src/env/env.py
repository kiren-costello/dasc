import networkx as nx
import numpy as np
import csv
import pandas as pd

from src.env.worker import Worker
from src.env.task import Task

from src.utils.get_abs_path import get_abs_path
from src.utils.road_network_tool import *


class CrowdsourcingEnv:
    def __init__(self, config_of_map, config_of_system, config_of_worker, config_of_task):
        self.map_file_path = get_abs_path() + "/data/map_data/"

        # Initialize Road Network
        self.roadnetwork = RoadNetworkTool(self.map_file_path,
                                           config_of_map['bbox'],
                                           network_type=config_of_map['network_type'])

        # Initialize Workers
        self.worker_list = []
        self.idle_worker_list = []

        self.time = 0

        self.task_list = []

        self.available_subtask_list = []
        self.recommended_subtask_list = []

        self.speed = config_of_worker['speed']
        self.worker_num = config_of_system['worker_num']
        self.maximum_distance_from_task = config_of_worker['maximum_distance_from_task']
        self.worker_movement_penalty = config_of_system['worker_movement_penalty']
        self.worker_num = config_of_system['worker_num']
        self.decision_time_step = config_of_system['decision_time_step']

        self.commission = 0

        self.task_time_discount = config_of_task['timeout_penalty']

        self.top_k_worker_nums = config_of_system['top_k_worker_nums']
        self.top_k_subtask_nums = config_of_system['top_k_subtask_nums']

        self.alpha = 0  # update from agent
        self.beta = 0  # update from agent

        self.update_task_num = 0
        self.task_upload_time_list = []

    """-----------------Time step update-----------------"""

    def update_worker(self):
        # Update the positions of workers,clear and update idle workers

        self.idle_worker_list = []

        for worker in self.worker_list:

            if worker.nest_rest_time > self.decision_time_step:
                worker.nest_rest_time -= self.decision_time_step

            else:
                worker.nest_rest_time = 0

                worker.assign_task = None
                self.idle_worker_list.append(worker)

    def update_task(self):

        for task in self.task_list[:]:

            if task.finish:
                self.commission += task.real_commission
                self.task_list.remove(task)
                continue

            task.time += self.decision_time_step

            for subtask in task.subtask_list:
                if not subtask.is_assigned and not subtask.is_finished:
                    subtask.last_finish_time -= self.decision_time_step
                    subtask.last_finish_expected_time -= self.decision_time_step

                if subtask.is_assigned:

                    if subtask.time_to_complete > self.decision_time_step:
                        subtask.time_to_complete -= self.decision_time_step

                    else:
                        subtask.is_finished = True
                        subtask.is_assigned = False
                        subtask.assigned_worker = None
                        task.is_assigned_subtask_list.remove(subtask.id)
                        task.finished_subtask_list.append(subtask.id)

                        task.subtask_finish_update(subtask.id)

    def update_available_subtask_list(self):
        self.available_subtask_list = []
        for task in self.task_list:
            for subtask_id in task.available_subtask_list:
                self.available_subtask_list.append(task.subtask_list[subtask_id])

    """-----------------Upload task and worker-----------------"""

    def update_task_time(self, time_update_csv):
        values = []
        with open(time_update_csv, 'r') as infile:
            time_update = csv.DictReader(infile)
            for row in time_update:
                # 提取 'Value' 列的值，并将其转换为整数
                values.append(int(row['Value']))
        self.task_upload_time_list = values

    def upload_task(self, task_csv, subtask_csv):
        if self.update_task_num < len(self.task_upload_time_list):
            if self.task_upload_time_list[self.update_task_num] <= self.time:
                taskcsv = pd.read_csv(task_csv)
                subtaskcsv = pd.read_csv(subtask_csv)

                while (self.update_task_num < len(self.task_upload_time_list) and
                       self.task_upload_time_list[self.update_task_num] <= self.time):

                    task_inf = taskcsv.loc[self.update_task_num]

                    subtask_indices = list(map(int, task_inf['Indices'].split(',')))
                    expected_time = int(task_inf['TypecodeSum1.3'])
                    deadline = int(task_inf['TypecodeSum2.5'])
                    remuneration = int(task_inf['TypecodeSum10'])
                    AdjacencyMatrix = task_inf['AdjacencyMatrix']

                    matrix = np.array([list(map(int, line.split(','))) for line in AdjacencyMatrix.strip().split('\n')])

                    subtask_information = []
                    for i in subtask_indices:
                        sub_inf = subtaskcsv.loc[i]

                        latitude = float(sub_inf['Latitude'])
                        longitude = float(sub_inf['Longitude'])

                        typecodes = list(map(int, sub_inf['Typecodes'].split(',')))

                        spendtime = int(sub_inf['RandomProperty'])

                        subtask_information.append([(latitude, longitude), typecodes, spendtime])

                    task = Task(self.update_task_num, subtask_information, matrix, deadline, expected_time,
                                remuneration, self.task_time_discount)

                    self.initialize_task_location(task)

                    self.task_list.append(task)

                    self.update_task_num += 1

    def upload_worker(self, worker_csv):
        with open(worker_csv, 'r') as infile:
            reader = csv.DictReader(infile)
            for index, row in enumerate(reader):
                latitude = float(row['Latitude'])
                longitude = float(row['Longitude'])
                typecodes = list(map(int, row['Typecodes'].split(',')))

                # 创建 Worker 对象
                worker = Worker(
                    worker_id=index,
                    worker_location=(latitude, longitude),
                    worker_skill_list=typecodes
                )
                self.worker_list.append(worker)

        self.initialize_worker_positions()

    def reset(self):
        # Reset the environment

        self.worker_list = []
        self.idle_worker_list = []

        self.time = 0

        self.task_list = []

        self.available_subtask_list = []
        self.recommended_subtask_list = []

        self.commission = 0

        self.update_task_num = 0
        self.task_upload_time_list = []

    """--------------------Initialize--------------------"""

    def initialize_task_location(self, task):
        # All uploaded tasks need to update the location

        for subtask in task.subtask_list:
            subtask.location = self.get_nearest_point_coordinates(subtask.location)

    def initialize_worker_positions(self):
        # Initialize the positions of workers

        for w in self.worker_list:
            w.location = self.get_nearest_point_coordinates(w.location)

    """-----------------Wrapper Function-----------------"""

    def get_nearest_point_coordinates(self, coord):
        point_id = self.roadnetwork.find_nearest_node(coord)
        point_inf = self.roadnetwork.graph.nodes[point_id]
        nearest_coord = (point_inf['y'], point_inf['x'])
        return nearest_coord

    """-----------------------Utility with KM----------------------------"""

    def get_observation(self):
        # Get the observation of the environment
        # graph_list = []
        graph_inf_list = []
        graph_matrix_list = []
        node_order_list = []
        top_k_index_subtask_list = []
        top_k_index_worker_list = []
        observations = [graph_inf_list, graph_matrix_list, node_order_list, top_k_index_subtask_list,
                        top_k_index_worker_list]
        worker_len = len(self.idle_worker_list)

        for task in self.task_list:

            for subtask_id in task.un_subtask_list:

                graph_inf, graph_matrix, node_order = task.subtask_list[subtask_id].get_graph_matrix()

                node_order_list.append(node_order)

                graph = []

                # graph.append(graph_inf)
                # graph.append(graph_matrix)
                # graph_list.append(graph)
                graph_inf_list.append(graph_inf)
                graph_matrix_list.append(graph_matrix)

                # top k subtask and top k worker

                distance_subtask = []
                distance_worker = []

                for i in range(len(self.available_subtask_list)):
                    # distance_of_subtask, _ = self.roadnetwork.find_the_shortest_path(
                    #     self.available_subtask_list[i].location,
                    #     task.subtask_list[subtask_id].location)
                    # distance_subtask.append(distance_of_subtask)
                    distance_of_subtask = haversine(self.available_subtask_list[i].location[0],
                                                    self.available_subtask_list[i].location[1],
                                                    task.subtask_list[subtask_id].location[0],
                                                    task.subtask_list[subtask_id].location[1])
                    distance_subtask.append(distance_of_subtask)

                if len(self.available_subtask_list) > self.top_k_subtask_nums:
                    top_k_index_subtask = sorted(range(len(distance_subtask)),
                                                 key=lambda index: distance_subtask[index])[
                                          :self.top_k_subtask_nums]
                else:
                    top_k_index_subtask = sorted(range(len(distance_subtask)),
                                                 key=lambda index: distance_subtask[index])

                # top_k_index_subtask_list.append(top_k_index_subtask)
                top_k_sub_embedding = []
                for k in top_k_index_subtask:
                    top_k_sub_embedding += self.available_subtask_list[k].get_inf_self()

                top_k_sub_embedding += (80 - len(top_k_sub_embedding)) * [0]

                top_k_index_subtask_list.append(top_k_sub_embedding)


                for i in range(len(self.idle_worker_list)):
                    # distance_of_worker, _ = self.roadnetwork.find_the_shortest_path(
                    #     task.subtask_list[subtask_id].location,
                    #     self.idle_worker_list[i].location)
                    # distance_worker.append(distance_of_worker)
                    distance_of_worker = haversine(task.subtask_list[subtask_id].location[0],
                                                   task.subtask_list[subtask_id].location[1],
                                                   self.idle_worker_list[i].location[0],
                                                   self.idle_worker_list[i].location[1])
                    distance_worker.append(distance_of_worker)

                if worker_len >= self.top_k_worker_nums:
                    top_k_index_worker = sorted(range(len(distance_worker)), key=lambda index: distance_worker[index])[
                                         :self.top_k_worker_nums]

                else:
                    top_k_index_worker = sorted(range(len(distance_worker)), key=lambda index: distance_worker[index])

                # top_k_index_worker_list.append(top_k_index_worker)
                top_k_sub_embedding = []

                for k in top_k_index_worker:
                    top_k_sub_embedding += flatten(
                        [self.idle_worker_list[k].location[0]-30.2,self.idle_worker_list[k].location[1]-103.6, self.idle_worker_list[k].skill_list])


                top_k_sub_embedding += (60 - len(top_k_sub_embedding)) * [0]

                top_k_index_worker_list.append(top_k_sub_embedding)

        return observations

    def get_state(self):
        state = []
        worker_state = []
        subtask_state = []

        for i in self.idle_worker_list:
            w_s = flatten([i.location, i.skill_list])
            worker_state.append(w_s)

        for task in self.task_list:
            for subtask in task.subtask_list:
                s_s = subtask.get_inf_self()
                subtask_state.append(s_s)

        state.append(worker_state)
        state.append(subtask_state)

        return state

    def step(self, actions, observations):
        if isinstance(actions, list):
            if not actions:
                return [], 0, [], 0, [], [], []

        actions_valid, adj_actions = self.action_valid(actions, observations)

        recommend_task = []
        i_action = 0

        for task in self.task_list:
            for subtask_id in task.un_subtask_list:

                if adj_actions[i_action]:
                    recommend_task.append(task.subtask_list[subtask_id])
                i_action += 1

        matching, bipartite_graph = self.get_bipartite_graph(recommend_task)

        self.update_after_km_matching(matching, recommend_task, bipartite_graph)

        ind_rewards, sum_ind = self.get_ind_reward(recommend_task, adj_actions, actions_valid)
        group_rewards, sum_group = self.get_group_reward(recommend_task, adj_actions, actions_valid)
        u1, u2, u3 = self.get_utility(recommend_task, adj_actions, ind_rewards)

        return ind_rewards, sum_ind, group_rewards, sum_group, u1, u2, u3

    def action_valid(self, actions, observations):
        actions_valid = [1] * len(actions)
        index = 0

        for i in range(len(self.task_list)):

            next_index = index + len(self.task_list[i].un_subtask_list)

            adj_matrix = observations[1][index]

            action_list = actions[index:next_index]

            invalid_index = find_invalid_recommendations(adj_matrix, action_list)
            for invalid in invalid_index:
                actions_valid[index + invalid] = 0

            index = next_index

        adjust_actions = [action * invalid for action, invalid in zip(actions, actions_valid)]

        return actions_valid, adjust_actions

    def get_ind_reward(self, task_list, adj_actions, valids):

        rewards = [0] * len(adj_actions)
        task_list_num = 0

        for i in range(len(adj_actions)):
            if adj_actions[i]:
                if task_list[task_list_num].is_assigned:
                    rewards[i] = task_list[task_list_num].ind_reward/20
                else:
                    rewards[i] = -1
                task_list_num += 1
            else:
                if valids[i]:
                    rewards[i] = 0
                else:
                    rewards[i] = -1

        return rewards, sum(rewards)

    def get_group_reward(self, task_list, adj_actions, valids):

        rewards = [0] * len(adj_actions)
        task_list_num = 0

        for i in range(len(adj_actions)):
            if adj_actions[i]:
                if task_list[task_list_num].is_assigned:
                    rewards[i] = task_list[task_list_num].group_reward
                else:
                    rewards[i] = 0  #-1
                task_list_num += 1
            # else:
            #     if valids[i]:
            #         rewards[i] = 0
            #     else:
            #         rewards[i] = -1

        return rewards, sum(rewards)

    def get_utility(self, task_list, adj_actions, ind_rewards):

        u1 = []
        u2 = []
        u3 = []
        task_list_num = 0
        for i in range(len(adj_actions)):
            if adj_actions[i]:
                if task_list[task_list_num].is_assigned:
                    if ind_rewards[i] > 0:
                        u1.append(task_list[task_list_num].u1)
                        u2.append(task_list[task_list_num].u2)
                        u3.append(task_list[task_list_num].u3)
                task_list_num += 1
                # u1.append(task_list[i].u1)
                # u2.append(task_list[i].u2)
                # u3.append(task_list[i].u3)
        return u1, u2, u3

    def calculate_utility(self, skill1, skill2, distance, t_d_time, e_d_time, alpha, beta):
        u_skill = skill1.count(1) / skill2.count(1)
        u_dist = (self.maximum_distance_from_task - distance) / self.maximum_distance_from_task
        u_time = 1/math.e * math.exp(1 - t_d_time / e_d_time)

        utility = alpha * beta * u_skill + alpha * (1 - beta) * u_dist + (1 - alpha) * u_time

        return utility, u_skill, u_dist, u_time

    def get_bipartite_graph(self, task_list):
        # Get the bipartite graph of self.idle_worker_list and task_list
        # todo 获取工人和任务的二部图
        bipartite_graph = nx.Graph()

        for subtask_order in range(len(task_list)):
            subtask_node = f"t{subtask_order}"
            bipartite_graph.add_node(subtask_node, bipartite=0)

        for worker_order in range(len(self.idle_worker_list)):
            worker_node = f"w{worker_order}"
            bipartite_graph.add_node(worker_node, bipartite=1)

            for subtask_order in range(len(task_list)):
                if self.idle_worker_list[worker_order].do_subtask(task_list[subtask_order]):
                    t_w_distance, _ = self.roadnetwork.find_the_shortest_path(
                        self.idle_worker_list[worker_order].location,
                        task_list[subtask_order].location)
                    # if t_w_distance < self.maximum_distance_from_task:
                    utility, u_skill, u_dist, u_time = self.calculate_utility(
                        task_list[subtask_order].skill_list,
                        self.idle_worker_list[worker_order].skill_list,
                        t_w_distance,
                        task_list[subtask_order].last_finish_time,
                        task_list[subtask_order].last_finish_time - task_list[
                            subtask_order].last_finish_expected_time,
                        self.alpha, self.beta)
                    subtask_node = f"t{subtask_order}"
                    bipartite_graph.add_edge(subtask_node, worker_node, weight=utility,
                                             u1=u_skill, u2=u_dist, u3=u_time)

        matching = nx.algorithms.matching.max_weight_matching(bipartite_graph, maxcardinality=True)
        return matching, bipartite_graph

    def update_after_km_matching(self, matching, task_list, bipartite_graph):

        matching = list(matching)
        matching_copy = matching[:]

        while matching_copy:

            for match_pair in matching:
                if match_pair[0][0] == 't':
                    subtask_node = match_pair[0]
                    worker_node = match_pair[1]

                    subtask_order = int(match_pair[0][1:])
                    worker_order = int(match_pair[1][1:])
                else:
                    subtask_node = match_pair[1]
                    worker_node = match_pair[0]

                    subtask_order = int(match_pair[1][1:])
                    worker_order = int(match_pair[0][1:])

                if task_list[subtask_order].id in task_list[subtask_order].task.available_subtask_list:
                    task_list[subtask_order].u1 = bipartite_graph[subtask_node][worker_node]['u1']
                    task_list[subtask_order].u2 = bipartite_graph[subtask_node][worker_node]['u2']
                    task_list[subtask_order].u3 = bipartite_graph[subtask_node][worker_node]['u3']

                    t_w_distance, _ = self.roadnetwork.find_the_shortest_path(
                        self.idle_worker_list[worker_order].location,
                        task_list[subtask_order].location)

                    task_list[subtask_order].task.after_matching_update_sub(
                        task_list[subtask_order].id, self.idle_worker_list[worker_order], t_w_distance, self.speed,
                        self.worker_movement_penalty)

                    self.idle_worker_list[worker_order].after_matching_update_worker(
                        task_list[subtask_order])
                    matching_copy.remove(match_pair)
            if matching == matching_copy:
                break
            matching = matching_copy[:]

    """------------------greedy algorithm------------------"""

    def greedy_algorithm(self):
        for worker in self.idle_worker_list:
            min_distance = float("inf")
            nearest_subtask = None
            for subtask in self.available_subtask_list:
                t_w_distance, _ = self.roadnetwork.find_the_shortest_path(worker.location, subtask.location)
                if t_w_distance < min_distance:
                    min_distance = t_w_distance
                    nearest_subtask = subtask

            if nearest_subtask is not None:
                self.available_subtask_list.remove(nearest_subtask)

                worker.after_matching_update_worker(nearest_subtask, min_distance, self.speed)

                nearest_subtask.task.after_matching_update_sub(nearest_subtask.id, worker)


def find_invalid_recommendations(adj_matrix, recommendations):
    invalid_nodes = []
    num_nodes = len(adj_matrix)
    for node in range(num_nodes):
        if recommendations[node] == 1:
            for predecessor in range(num_nodes):
                if adj_matrix[predecessor][node] == 1 and recommendations[predecessor] == 0:
                    invalid_nodes.append(node)
                    break

    return invalid_nodes


def flatten(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list) or isinstance(element, tuple):
            flat_list.extend(flatten(element))
        else:
            flat_list.append(element)
    return flat_list


def haversine(lon1, lat1, lon2, lat2):
    # R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    # c = 2 * math.asin(math.sqrt(a))
    return a
