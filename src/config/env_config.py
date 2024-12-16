worker_config = {
    "maximum_distance_from_task": 5000,
    "speed": 10,
}

task_config = {
    "timeout_penalty": 0.02,
}

system_config = {
    "maximum_distance_from_task": 5000,
    "speed": 10,
    "decision_time_step": 5,
    "worker_num": 1000,
    "worker_movement_penalty": 0.0001,
    "top_k_worker_nums": 5,
    "top_k_subtask_nums": 5,
}
