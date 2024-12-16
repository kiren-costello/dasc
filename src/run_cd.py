from tqdm import tqdm
import random
import pandas as pd

from src.env.env import CrowdsourcingEnv
from src.net.agent import Agent

from src.config.map_config.map_chengdu import map_config
from src.config.env_config import worker_config, system_config, task_config
from src.config.model_config_cd2 import agent_config, policy_config, value_config, meta_gradient_hyperparameters, \
    train_config

env = CrowdsourcingEnv(map_config, system_config, worker_config, task_config)
agent = Agent(agent_config, policy_config, value_config, meta_gradient_hyperparameters)

# agent.init_net()
# agent.save_model(0)
qwe = 0
agent.read_model(qwe)
ccc = 0
with tqdm(total=train_config["episode"], desc='meta update') as pbar:
    for i in range(train_config["episode"]):

        """-----------------start-----------------"""
        """Step.1 upload"""
        env.alpha = agent.alpha
        env.beta = agent.beta

        worker_csv = '../data/cd_worker/worker300.csv'
        subtask_csv = '../data/cd_subtask.csv'

        random_number = random.randint(0, 999)

        time_update_csv = '../data/time/60/upload_time' + str(random_number) + '.csv'
        task_csv = '../data/cd_task/60/task' + str(random_number) + '.csv'

        env.upload_worker(worker_csv)
        env.update_task_time(time_update_csv)
        env.upload_task(task_csv, subtask_csv)

        env.update_worker()
        env.update_task()
        env.update_available_subtask_list()

        next_state = env.get_state()

        done = 0

        env.time += env.decision_time_step
        for jkl in range(1000):
            pbar.set_postfix({'jkl': '%d' % jkl})
            if done:
                break
            # while not done:
            """Step.2 get state and observations"""

            state = next_state
            observations = env.get_observation()

            """Step.3 get action"""
            # print(observations)
            action = agent.select_action(observations)

            """Step.3 take action and get rewards"""
            try:
                ind_rewards, sum_ind, group_rewards, sum_group, u1, u2, u3 = env.step(action, observations)
            except:
                ccc = 1

                break

            # print(ind_rewards)
            """Step.4 upload task and update state"""
            env.time += env.decision_time_step

            # upload and update
            env.upload_task(task_csv, subtask_csv)

            env.update_worker()
            env.update_task()
            env.update_available_subtask_list()

            next_state = env.get_state()
            """Step.5 store the transition"""
            agent.store_transition(observations, action, group_rewards, ind_rewards, sum_group, state, next_state, done,
                                   u1,
                                   u2, u3)

            if not env.task_list and env.update_task_num == len(env.task_upload_time_list):
                done = 1

            if env.time >= 10000:
                break

        if ccc == 1:
            agent.buffer.reset()
            env.reset()
            ccc = 0
            continue

        file_path = '../models/CDtrain_csv/commission.csv'
        df = pd.read_csv(file_path)
        new_row = {
            'episode': i + 1 + qwe,
            'commission': env.commission
        }
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(file_path, index=False)

        """Step.6 update"""
        pbar.set_postfix({'commission': '%d' % env.commission})

        agent.meta_gradient_update_group_strategy()
        # pbar.set_postfix({'iteration': '%d' % j,
        #                   'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
        file_path = '../models/CDtrain_csv/alpha_beta.csv'
        df = pd.read_csv(file_path)
        new_row = {
            'episode': i + 1 + qwe,
            'alpha': agent.alpha,
            'beta': agent.beta,
        }
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(file_path, index=False)

        agent.buffer.reset()
        env.reset()

        agent.save_model(i + 1 + qwe)

        agent.xi = agent.xi * 0.999
