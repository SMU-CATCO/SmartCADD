import bocas
import os
from datetime import datetime
import tensorflow as tf
import termcolor
import resource
import loader
import models
from utils import get_n_trainable_weights

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def get_model(config, n_classes):
    return models.create_model(size=config.size, n_classes=n_classes)


def get_name(config):
    # Resume the training of a model
    if config.model_name is not None:
        return config.model_name

    now = datetime.now()
    return f"{config.size}_{now.strftime('%m_%d_%y_%H_%M')}_{config.version}"


def get_datasets(config):
    train_ds, info = loader.load(
        split="train",
        batch_size=config.batch_size,
        augment=True,
        return_info=True,
        data_dir=config.data_dir,
    )

    # Get the number of examples in the training set
    n_training_samples = info.splits["train"].num_examples

    # Get the number of classes
    n_classes = info.features["label"].num_classes

    # Load datasets
    val_ds = loader.load(
        split="validation",
        batch_size=config.batch_size,
        augment=True,
        data_dir=config.data_dir,
    )
    eval_ds = loader.load(
        split="test",
        batch_size=config.batch_size,
        augment=False,
        data_dir=config.data_dir,
    )

    return train_ds, val_ds, eval_ds, n_training_samples, n_classes


def get_optimizer(config, n_training_samples):
    first_decay_steps = n_training_samples // config.batch_size
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        config.initial_learning_rate,
        first_decay_steps=first_decay_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.0,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    return optimizer


def get_callbacks(config):
    return [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.BackupAndRestore(
            os.path.join("./models", config.version, config.name, "backup")
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("./models", config.version, config.name, "ckpt"),
            save_best_only=True,
            monitor=config.monitor,
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("./models", config.version, config.name, "logs"),
            profile_batch=(5, 6),
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=config.monitor,
            mode="min",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-3,
        ),
    ]


def train_and_evaluate_model(config):
    # Get the datasets
    train_ds, val_ds, eval_ds, n_training_samples, n_classes = get_datasets(config)

    # Get the model
    model = get_model(config, n_classes)

    # Add model metrics to the config
    config.trainable_parameters = get_n_trainable_weights(model)

    # Show the model summary
    model.summary()

    # Compile the model
    model.compile(
        optimizer=get_optimizer(config, n_training_samples),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
        run_eagerly=config.run_eagerly,
        jit_compile=config.jit_compile,
    )

    # Get the callbacks
    callbacks = get_callbacks(config)

    # Fit the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=config.verbose,
    )

    # Evaluate the model
    metrics = model.evaluate(eval_ds, return_dict=True, verbose=config.verbose)

    return history, metrics


def run(config):
    name = get_name(config)
    config.name = name
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))
    termcolor.cprint(
        termcolor.colored(f"Training model: {name}", "green", attrs=["bold"])
    )
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))

    if config.n_gpus > 1:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        strategy = tf.distribute.MirroredStrategy()
        number_of_devices = strategy.num_replicas_in_sync

        with strategy.scope():
            if number_of_devices != config.n_gpus:
                raise RuntimeError(
                    f"Expected {config.n_gpus} GPUs, but "
                    f"found {number_of_devices} GPUs."
                )

            history, metrics = train_and_evaluate_model(config)

    else:
        history, metrics = train_and_evaluate_model(config)

    return bocas.Result(
        name=name,
        config=config,
        artifacts=[
            bocas.artifacts.KerasHistory(history, name="history"),
            bocas.artifacts.Metrics(metrics, name="metrics"),
        ],
    )
