import torch.optim

from src.net.network import *
from src.utils.get_abs_path import get_abs_path
from src.utils.tensor_convert import *


class Buffer:
    def __init__(self):
        self.policy_memory = {"graph_inf": [],
                              "graph_matrix": [],
                              "node_order": [],
                              "worker_inf": [],
                              "subtask_inf": [],
                              "actions": [],
                              "ind_reward": [],
                              "group_reward": [],
                              "u_1": [],
                              "u_2": [],
                              "u_3": [],
                              }
        self.value_memory = {"state_workers": [],
                             "state_subtask": [],
                             "next_state_worker": [],
                             "next_state_subtask": [],
                             "global_rewards": [],
                             "dones": [],
                             }
        self.position = 0

    def reset(self):
        self.policy_memory = {"graph_inf": [],
                              "graph_matrix": [],
                              "node_order": [],
                              "worker_inf": [],
                              "subtask_inf": [],
                              "actions": [],
                              "ind_reward": [],
                              "group_reward": [],
                              "u_1": [],
                              "u_2": [],
                              "u_3": [],
                              }
        self.value_memory = {"state_workers": [],
                             "state_subtask": [],
                             "next_state_worker": [],
                             "next_state_subtask": [],
                             "global_rewards": [],
                             "dones": [],
                             }
        self.position = 0

    def store_transition(self, graph_inf, graph_matrix, node_order, worker_inf, subtask_inf, action, ind_reward,
                         group_reward,
                         u_1, u_2, u_3,
                         state_workers, state_subtask, next_state_worker, next_state_subtask, global_rewards, done):
        self.policy_memory["graph_inf"].append(graph_inf)
        self.policy_memory["graph_matrix"].append(graph_matrix)
        self.policy_memory["node_order"].append(node_order)
        self.policy_memory["worker_inf"].append(worker_inf)
        self.policy_memory["subtask_inf"].append(subtask_inf)
        self.policy_memory["actions"].append(action)
        self.policy_memory["ind_reward"].append(ind_reward)
        self.policy_memory["group_reward"].append(group_reward)

        self.policy_memory["u_1"].append(u_1)
        self.policy_memory["u_2"].append(u_2)
        self.policy_memory["u_3"].append(u_3)

        self.value_memory["state_workers"].append(state_workers)
        self.value_memory["state_subtask"].append(state_subtask)
        self.value_memory["next_state_worker"].append(next_state_worker)
        self.value_memory["next_state_subtask"].append(next_state_subtask)
        self.value_memory["global_rewards"].append(global_rewards)
        self.value_memory["dones"].append(done)

        self.position += 1

    def sample(self, batch_size):
        # Ensure the batch size does not exceed the buffer length
        if batch_size > self.position:
            return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        start_idx = np.random.randint(0, self.position - batch_size + 1)

        # Extract the segment from both buffers
        graph_inf_list = self.policy_memory["graph_inf"][start_idx:start_idx + batch_size]
        graph_matrix_list = self.policy_memory["graph_matrix"][start_idx:start_idx + batch_size]
        node_order_list = self.policy_memory["node_order"][start_idx:start_idx + batch_size]
        worker_inf_list = self.policy_memory["worker_inf"][start_idx:start_idx + batch_size]
        subtask_inf_list = self.policy_memory["subtask_inf"][start_idx:start_idx + batch_size]
        actions_list = self.policy_memory["actions"][start_idx:start_idx + batch_size]
        ind_reward_list = self.policy_memory["ind_reward"][start_idx:start_idx + batch_size]
        group_reward_list = self.policy_memory["group_reward"][start_idx:start_idx + batch_size]

        u_1_list = self.policy_memory["u_1"][start_idx:start_idx + batch_size]
        u_2_list = self.policy_memory["u_2"][start_idx:start_idx + batch_size]
        u_3_list = self.policy_memory["u_3"][start_idx:start_idx + batch_size]

        state_workers_list = self.value_memory["state_workers"][start_idx:start_idx + batch_size]
        state_subtask_list = self.value_memory["state_subtask"][start_idx:start_idx + batch_size]
        next_state_worker_list = self.value_memory["next_state_worker"][start_idx:start_idx + batch_size]
        next_state_subtask_list = self.value_memory["next_state_subtask"][start_idx:start_idx + batch_size]
        global_rewards_list = self.value_memory["global_rewards"][start_idx:start_idx + batch_size]
        dones_list = self.value_memory["dones"][start_idx:start_idx + batch_size]

        return (graph_inf_list, graph_matrix_list, node_order_list, worker_inf_list, subtask_inf_list,
                actions_list, ind_reward_list, group_reward_list,
                u_1_list, u_2_list, u_3_list,
                state_workers_list, state_subtask_list,
                next_state_worker_list, next_state_subtask_list,
                global_rewards_list, dones_list)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_array = np.array(advantage_list, dtype=np.float32)
    return torch.tensor(advantage_array, dtype=torch.float)


class Agent:
    def __init__(self, config_agent, config_policy, config_value, config_meta):
        self.config_policy = config_policy
        self.policy_learning_rate = config_agent["policy_learning_rate"]
        self.value_learning_rate = config_agent["value_learning_rate"]
        self.gamma = config_agent["gamma"]
        self.lmbda = config_agent["lmbda"]
        self.epsilon = config_agent["epsilon"]
        self.iteration = config_agent["iteration"]
        self.mini_batch_size = config_agent["mini_batch_size"]

        self.meta_config = config_meta
        self.config_model_path = config_agent["model_save_path"]
        self.model_save_path = get_abs_path() + self.config_model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.buffer = Buffer()

        self.critic = ValueNet(config_value).to(self.device)
        self.group_actor = PolicyNet(config_policy).to(self.device)
        self.individual_actor = PolicyNet(config_policy).to(self.device)

        self.old_group_actor = PolicyNet(config_policy).to(self.device)

        self.meta_optimizer = torch.optim.SGD(self.group_actor.parameters(), lr=config_meta["strategy_update_step"])
        self.individual_optimizer = torch.optim.SGD(self.individual_actor.parameters(), lr=self.policy_learning_rate)
        self.group_optimizer = torch.optim.SGD(self.group_actor.parameters(), lr=self.policy_learning_rate)
        self.value_optimizer = torch.optim.SGD(self.critic.parameters(), lr=self.value_learning_rate)

        self.alpha = config_meta["default_alpha"]
        self.beta = config_meta["default_beta"]

        self.utility_learning_rate = config_meta["utility_learning_rate"]
        self.soft_update_coefficient = config_meta["soft_update_coefficient"]

        self.xi = config_meta["xi"]

    def init_net(self):
        def normal_init(m):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.group_actor.apply(normal_init)
        self.individual_actor.apply(normal_init)
        self.critic.apply(normal_init)

    def save_model(self, model_id):
        torch.save(self.group_actor.state_dict(), self.model_save_path + "group_actor_" + str(model_id) + ".pth")
        torch.save(self.individual_actor.state_dict(),
                   self.model_save_path + "individual_actor_" + str(model_id) + ".pth")
        torch.save(self.critic.state_dict(), self.model_save_path + "critic_" + str(model_id) + ".pth")

    def read_model(self, model_id):
        self.group_actor.load_state_dict(torch.load(self.model_save_path + "group_actor_" + str(model_id) + ".pth"))
        self.individual_actor.load_state_dict(
            torch.load(self.model_save_path + "individual_actor_" + str(model_id) + ".pth"))
        self.critic.load_state_dict(torch.load(self.model_save_path + "critic_" + str(model_id) + ".pth"))

    def process_observation(self, observation):
        graph_inf = pad_and_tensorize_embeddings(observation[0]).to(self.device)
        graph_matrix, mask = pad_and_tensorize_adjacency_matrices(observation[1])
        node_order = observation[2]
        worker_inf = torch.tensor(observation[3], dtype=torch.float).to(self.device)
        subtask_inf = torch.tensor(observation[4], dtype=torch.float).to(self.device)

        return graph_inf, graph_matrix.to(self.device), mask.to(self.device), node_order, worker_inf, subtask_inf

    def select_action(self, observation):
        graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf = self.process_observation(observation)
        if not graph_inf.numel():
            return []
        probs = self.group_actor(graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf)

        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().cpu()
        return action.numpy()  # such as [1 0 1 0 1 0]

    def select_action_ind(self, observation):
        graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf = self.process_observation(observation)
        if not graph_inf.numel():
            return []
        probs = self.individual_actor(graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().cpu()
        return action.numpy()

    def select_max_action(self, observation):
        graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf = self.process_observation(observation)

        probs = self.group_actor(graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf)

        action = torch.argmax(probs, dim=-1).cpu()
        return action.numpy()

    def select_max_action_ind(self, observation):
        graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf = self.process_observation(observation)

        probs = self.group_actor(graph_inf, graph_matrix, mask, node_order, worker_inf, subtask_inf)
        action = torch.argmax(probs, dim=-1).cpu()
        return action.numpy()

    def store_transition(self, observation, action, group_reward, ind_reward, sum_group, state, next_state, done, a, b,
                         c):
        # todo 计算全局奖励
        global_reward = 0
        self.buffer.store_transition(observation[0], observation[1], observation[2], observation[3], observation[4],
                                     action, ind_reward, group_reward,
                                     a, b, c,
                                     state[0], state[1], next_state[0], next_state[1],
                                     sum_group, done)

    def meta_gradient_update_group_strategy(self):
        """meta update group strategy"""

        self.old_group_actor.load_state_dict(self.group_actor.state_dict())

        alpha = self.alpha
        beta = self.beta

        def utility_function(al, be, u1, u2, u3):
            return al * be * u1 + al * (1 - be) * u2 + (1 - al) * u3

        for _ in range(self.iteration):
            (graph_inf_list, graph_matrix_list, node_order_list, worker_inf_list, subtask_inf_list,
             actions_list, ind_reward_list, group_reward_list,
             u_1_list, u_2_list, u_3_list,
             state_workers_list, state_subtask_list,
             next_state_worker_list, next_state_subtask_list,
             global_rewards_list, dones_list) = self.buffer.sample(self.mini_batch_size)

            if graph_inf_list == []:
                continue

            """Step.1 Convert state transition_dict to tensor"""

            state_workers_list = pad_and_tensorize_embeddings(state_workers_list).to(self.device)
            state_subtask_list = pad_and_tensorize_embeddings(state_subtask_list).to(self.device)
            next_state_worker_list = pad_and_tensorize_embeddings(next_state_worker_list).to(self.device)
            next_state_subtask_list = pad_and_tensorize_embeddings(next_state_subtask_list).to(self.device)
            if not state_subtask_list.numel() or not next_state_subtask_list.numel() or not state_workers_list.numel() or not next_state_worker_list.numel():
                continue
            global_reward = torch.tensor(global_rewards_list, dtype=torch.float).view(-1, 1).to(self.device)
            dones = torch.tensor(dones_list, dtype=torch.float).view(-1, 1).to(self.device)

            td_target = global_reward + self.gamma * self.critic(next_state_worker_list, next_state_subtask_list) * (
                    1 - dones)
            td_delta = td_target - self.critic(state_workers_list, state_subtask_list)

            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

            for i in range(self.mini_batch_size):
                """Step.2 Convert observation transition_dict to tensor"""

                ind_reward = torch.tensor(ind_reward_list[i], dtype=torch.float).view(-1, 1).to(self.device)

                if not ind_reward.numel():
                    continue

                group_reward = torch.tensor(group_reward_list[i], dtype=torch.float).view(-1, 1).to(self.device)
                actions = torch.tensor(actions_list[i]).view(-1, 1).to(self.device)
                graph_inf = pad_and_tensorize_embeddings(graph_inf_list[i]).to(self.device)
                graph_matrix, graph_mask = pad_and_tensorize_adjacency_matrices(graph_matrix_list[i])
                graph_matrix = graph_matrix.to(self.device)
                graph_mask = graph_mask.to(self.device)
                node_order = node_order_list[i]
                worker_inf = torch.tensor(worker_inf_list[i], dtype=torch.float).to(self.device)
                subtask_inf = torch.tensor(subtask_inf_list[i], dtype=torch.float).to(self.device)

                ind_agent_advantage = (ind_reward / global_reward[i]) * advantage[i].detach()
                group_agent_advantage = (group_reward / global_reward[i]) * advantage[i].detach()

                # if len(actions) > 0:
                #     self.soft_update_coefficient = 1 / self.mini_batch_size / self.iteration / len(actions)
                # else:
                #     self.soft_update_coefficient = 0.0000000001
                meta = True
                if meta and not torch.all(torch.isnan(ind_agent_advantage)):
                    """Step.3 meta gradient of group strategy"""

                    ind_log_probs = torch.log(
                        self.individual_actor(graph_inf, graph_matrix, graph_mask, node_order, worker_inf,
                                              subtask_inf).gather(1,
                                                                  actions) + 1e-9)

                    meta_group_log_probs = torch.log(
                        self.group_actor(graph_inf, graph_matrix, graph_mask, node_order, worker_inf,
                                         subtask_inf).gather(1, actions) + 1e-9)

                    meta_ratio = torch.exp(ind_log_probs - meta_group_log_probs)

                    meta_ind_actor_loss = torch.mean(
                        -torch.min(meta_ratio * ind_agent_advantage, torch.clamp(meta_ratio, 1 - self.epsilon,
                                                                                 1 + self.epsilon) * ind_agent_advantage))

                    meta_grads_ind_actor_net = torch.autograd.grad(outputs=meta_ind_actor_loss,
                                                                   inputs=list(self.individual_actor.parameters()),
                                                                   grad_outputs=torch.ones_like(meta_ind_actor_loss),
                                                                   create_graph=True,
                                                                   retain_graph=True)

                    meta_second_grads = []
                    for meta_grad_ind_actor_net in meta_grads_ind_actor_net:
                        meta_grads_ind_group = torch.autograd.grad(outputs=meta_grad_ind_actor_net,
                                                                   inputs=list(self.group_actor.parameters()),
                                                                   grad_outputs=torch.ones_like(
                                                                       meta_grad_ind_actor_net),
                                                                   retain_graph=True)
                        meta_second_grads.append(meta_grads_ind_group)

                    """Step.4 build the loss of individual"""

                    self.individual_optimizer.zero_grad(set_to_none=True)
                    self.meta_optimizer.zero_grad(set_to_none=True)

                    group_log_probs = meta_group_log_probs

                    ind_ratio = torch.exp(ind_log_probs - group_log_probs)

                    ind_actor_loss = torch.mean(-torch.min(ind_ratio * ind_agent_advantage,
                                                           torch.clamp(ind_ratio, 1 - self.epsilon,
                                                                       1 + self.epsilon) * ind_agent_advantage))

                    ind_actor_loss.backward()

                    """Step.5 Combining meta gradients"""

                    for group_param, ind_param, meta_grads in (
                            zip(self.group_actor.parameters(), self.individual_actor.parameters(),
                                zip(*meta_second_grads))):
                        group_param.grad = -self.policy_learning_rate / 2 * self.soft_update_coefficient * ind_param.grad * sum(
                            meta_grads) * self.xi

                    # for group_param, ind_param in zip(self.group_actor.parameters(), self.individual_actor.parameters(),
                    #                                   zip(*meta_second_grads)):
                    #     group_param.grad = self.soft_update_coefficient * ind_param.grad * sum(meta_second_grads)
                    for ind_param in self.individual_actor.parameters():
                        ind_param.grad = ind_param.grad * self.soft_update_coefficient

                    nn.utils.clip_grad_norm_(self.group_actor.parameters(), max_norm=1 - 1e-6)
                    nn.utils.clip_grad_norm_(self.individual_actor.parameters(), max_norm=1 - 1e-6)
                    for group_param in self.group_actor.parameters():
                        if torch.all(torch.isnan(group_param.grad)):
                            if group_param.grad.dim() == 0:
                                group_param.grad = torch.tensor(1e-9, dtype=torch.float, device=self.device)
                            else:
                                group_param.grad[:] = 1e-9
                            # group_param.grad = torch.tensor(0, dtype=torch.float, device=self.device)
                    for ind_param in self.individual_actor.parameters():
                        if torch.all(torch.isnan(ind_param.grad)):
                            if ind_param.grad.dim() == 0:
                                ind_param.grad = torch.tensor(1e-9, dtype=torch.float, device=self.device)
                            else:
                                ind_param.grad[:] = 1e-9
                            # ind_param.grad = torch.tensor(0, dtype=torch.float, device=self.device)
                    self.meta_optimizer.step()
                    self.individual_optimizer.step()

                """Step.6 update group strategy"""
                if not torch.all(torch.isnan(group_agent_advantage)):
                    self.group_optimizer.zero_grad(set_to_none=True)
                    group_log_probs = torch.log(
                        self.group_actor(graph_inf, graph_matrix, graph_mask, node_order, worker_inf,
                                         subtask_inf).gather(1, actions))
                    group_old_log_probs = torch.log(
                        self.old_group_actor(graph_inf, graph_matrix, graph_mask, node_order, worker_inf,
                                             subtask_inf).gather(1, actions)).detach()

                    group_ratio = torch.exp(group_log_probs - group_old_log_probs)

                    group_actor_loss = torch.mean(-torch.min(group_ratio * group_agent_advantage,
                                                             torch.clamp(group_ratio, 1 - self.epsilon,
                                                                         1 + self.epsilon) * group_agent_advantage))
                    group_actor_loss = group_actor_loss * self.soft_update_coefficient
                    group_actor_loss.backward()
                    for group_param in self.group_actor.parameters():
                        # group_param.grad = group_param.grad * self.soft_update_coefficient
                        if torch.all(torch.isnan(group_param.grad)):
                            if group_param.grad.dim() == 0:
                                group_param.grad = torch.tensor(0, dtype=torch.float, device=self.device)
                            else:
                                group_param.grad[:] = 0
                            # group_param.grad = torch.tensor(0, dtype=torch.float, device=self.device)
                    nn.utils.clip_grad_norm_(self.group_actor.parameters(), max_norm=1 - 1e-6)
                    self.group_optimizer.step()

                if meta and not torch.all(torch.isnan(ind_agent_advantage)):
                    """Step.7 judgement update the utility """
                    r_mask = ind_reward > 0

                    if not r_mask.any():
                        continue

                    action_mask = actions[r_mask.view(-1)]
                    graph_inf_mask = graph_inf[r_mask.view(-1)]
                    graph_mask_mask = graph_mask[r_mask.view(-1)]
                    graph_matrix_mask = graph_matrix[r_mask.view(-1)]
                    # node_order_mask = node_order[r_mask.view(-1)]

                    r_mask_n = r_mask.view(-1).cpu().numpy()
                    node_order_mask = [node_order[i] for i in range(len(node_order)) if r_mask_n[i]]

                    worker_inf_mask = worker_inf[r_mask.view(-1)]
                    subtask_inf_mask = subtask_inf[r_mask.view(-1)]
                    ind_agent_advantage_mask = ind_agent_advantage[r_mask.view(-1)].requires_grad_()

                    u_1 = torch.tensor(u_1_list[i], dtype=torch.float).view(-1, 1).to(self.device)
                    u_2 = torch.tensor(u_2_list[i], dtype=torch.float).view(-1, 1).to(self.device)
                    u_3 = torch.tensor(u_3_list[i], dtype=torch.float).view(-1, 1).to(self.device)

                    """Step.8 meta gradient of utility"""

                    ind_log_mask = torch.log(
                        self.individual_actor(graph_inf_mask, graph_matrix_mask, graph_mask_mask, node_order_mask,
                                              worker_inf_mask,
                                              subtask_inf_mask).gather(1, action_mask) + 1e-9)

                    group_log_mask = torch.log(
                        self.group_actor(graph_inf_mask, graph_matrix_mask, graph_mask_mask, node_order_mask,
                                         worker_inf_mask,
                                         subtask_inf_mask).gather(1, action_mask) + 1e-9).detach()

                    ratio_mask = torch.exp(ind_log_mask - group_log_mask)

                    ind_actor_loss_mask = torch.mean(-torch.min(ratio_mask * ind_agent_advantage_mask,
                                                                torch.clamp(ratio_mask, 1 - self.epsilon,
                                                                            1 + self.epsilon) * ind_agent_advantage_mask))

                    meta_grads_advantage = torch.autograd.grad(outputs=ind_actor_loss_mask,
                                                               inputs=ind_agent_advantage_mask,
                                                               grad_outputs=torch.ones_like(ind_actor_loss_mask),
                                                               create_graph=True,
                                                               retain_graph=True)
                    meta_grads_advantage = meta_grads_advantage[0]
                    meta_second_grads_ind = torch.autograd.grad(outputs=meta_grads_advantage,
                                                                inputs=list(self.individual_actor.parameters()),
                                                                grad_outputs=torch.ones_like(meta_grads_advantage),
                                                                retain_graph=True)

                    self.individual_optimizer.zero_grad(set_to_none=True)
                    ind_actor_loss_mask.backward()

                    """Step.10 Combining meta gradients of utility"""
                    u_meta = 0

                    for ind_param, meta_second_grad_in in zip(self.individual_actor.parameters(),
                                                              meta_second_grads_ind):
                        grad_sum = sum(ind_param.grad * meta_second_grad_in)
                        while grad_sum.shape != torch.Size([]):
                            grad_sum = sum(grad_sum)
                        u_meta += grad_sum
                    meta_alpha = torch.tensor(alpha, dtype=torch.float, requires_grad=True, device=self.device)
                    meta_beta = torch.tensor(beta, dtype=torch.float, requires_grad=True, device=self.device)

                    result = utility_function(meta_alpha, meta_beta, u_1, u_2, u_3)

                    result = result.sum()

                    result.backward()
                    # meta_alpha.grad.data.clamp(-1.0, 1.0)
                    # meta_beta.grad.data.clamp(-1.0, 1.0)
                    # u_meta = torch.tensor(u_meta, dtype=torch.float, device=self.device)
                    # if torch.is_tensor(u_meta):
                    # u_meta = torch.clamp(u_meta, min=-1000.0, max=1000.0)
                    if not torch.isnan(u_meta):
                        alpha_grad = meta_alpha.grad * u_meta
                        beta_grad = meta_beta.grad * u_meta

                        self.alpha -= -self.policy_learning_rate / 2 * self.utility_learning_rate * alpha_grad.item() * self.soft_update_coefficient
                        self.beta -= -self.policy_learning_rate / 2 * self.utility_learning_rate * beta_grad.item() * self.soft_update_coefficient

                        self.alpha = max(0, min(1, self.alpha))
                        self.beta = max(0, min(1, self.beta))

            """Step.11 update value function"""

            value_loss = torch.mean(F.mse_loss(self.critic(state_workers_list, state_subtask_list), td_target.detach()))
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1 - 1e-6)
            self.value_optimizer.step()
