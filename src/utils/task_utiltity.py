import math


class Utility:
    def __init__(self,config_of_worker):
        self.max_distance = config_of_worker["maximum_distance_from_task"]

    def skill_compatibility_utility(self, skill1, skill2):
        u_skill = skill1.count(1) / skill2.count(1)
        return u_skill

    def distance_utility(self, distance):
        u_dist = (self.max_distance - distance) / self.max_distance
        return u_dist

    def time_remaining_utility(self, t_d_time, e_d_time):
        u_time = math.e * math.exp(1 - t_d_time / e_d_time)
        return u_time

    def matching_utility(self, v_1, v_2, v_3, u_skill, u_dist, u_time):
        return v_1 * u_skill + v_2 * u_dist + v_3 * u_time