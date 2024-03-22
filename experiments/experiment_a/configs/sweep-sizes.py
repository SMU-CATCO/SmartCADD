import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # General info
    config.computer = "superpod"
    config.n_gpus = 8
    if config.computer in ["superpod", "superpod-test"]:
        config.data_dir = "/data"
    elif config.computer == "local":
        config.data_dir = "/Volumes/harper1tb"
    else:
        raise ValueError(f"Unknown computer {config.computer}")

    # Model info
    config.model_type = "Example"
    config.size = bocas.Sweep([1, 2, 3])

    # Optimization info
    config.batch_size = 128 * max(config.n_gpus, 1)
    config.epochs = 100
    config.initial_learning_rate = 1e-4
    config.monitor = "val_loss"

    # Meta info
    config.version = "v0.1"
    config.verbose = 2
    config.run_eagerly = False
    config.jit_compile = False  # NOTE: tf.ragged is not supported by XLA

    # Resume training
    config.model_name = None  # Change to model name to resume training

    # If config.name exists, throw an error
    if hasattr(config, "name"):
        raise ValueError(
            "config.name is reserved and will be added at runtime. "
            "This enables the name to be unique for each run across sweeps. "
            "Please change the name of the config element."
        )

    return config
