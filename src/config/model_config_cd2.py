agent_config = {
    "policy_learning_rate": 0.001,
    "value_learning_rate": 0.001,
    "gamma": 0.99,
    "lmbda": 0.95,
    "epsilon": 0.2,

    "iteration": 10,
    "mini_batch_size": 64,
    "model_save_path": "/models/CD02/"

}

policy_config = {
    "GAT_input_dim": 16,
    "GAT_hidden_dim": 16,
    "GAT_output_dim": 16,
    "GAT_dropout": 0.1,
    "GAT_alpha": 0.2,
    "GAT_head": 8,

    "encoder_input_dim": 156,
    "mlp_hidden_dim1": 78,
    "mlp_hidden_dim2": 36,
    "mlp_output_dim": 2,
    "max_nodes": 15,
}

value_config = {
    "worker_embedding_dim": 12,
    "worker_hidden_dim": 12,
    "worker_output_dim": 12,

    "subtask_embedding_dim": 16,
    "subtask_hidden_dim": 16,
    "subtask_output_dim": 16,

    "encoder_input_dim": 28,
    "mlp_hidden_dim1": 16,
    "mlp_hidden_dim2": 4,
    "mlp_output_dim": 1,
}

meta_gradient_hyperparameters = {
    # Other strategies learning rate
    "xi": 0.6,

    "utility_learning_rate": 0.1,
    "strategy_update_step": 0.001,
    "default_alpha": 0.2,
    "default_beta": 0.2,
    "soft_update_coefficient": 0.001,
}

train_config = {
    "episode": 1000,
    "iteration": 1,
}
