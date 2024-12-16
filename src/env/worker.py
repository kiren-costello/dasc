class Worker:
    def __init__(self, worker_id, worker_location, worker_skill_list):
        self.id = worker_id
        self.location = worker_location
        self.skill_list = worker_skill_list

        self.nest_rest_time = 0

        self.assign_task = None

    def do_subtask(self, subtask):
        # Determine whether the worker can complete the task
        return all(self.skill_list[i] == 1 for i in range(len(self.skill_list))
                   if subtask.skill_list[i] == 1)

    def after_matching_update_worker(self, subtask):
        self.location = subtask.location
        self.assign_task = subtask
        self.nest_rest_time = subtask.time_to_complete


