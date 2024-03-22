import tensorflow_datasets as tfds
import tensorflow as tf
import resource
import os

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


# Create data pipeline
def prepare_dataset(ds, batch_size=16, augment=True, shuffle=True, deterministic=False):
    if shuffle:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )

    ds = (
        ds.batch(batch_size)
        .map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


def load(split, batch_size, augment, data_dir, return_info=False):
    if split not in ["train", "validation", "test"]:
        raise ValueError(
            f"expected split to be train, validation, or test, got {split}."
        )

    dataset_name = "cifar10"
    if dataset_name == "cifar10":
        if split == "validation":
            split = "train"
    dataset, info = tfds.load(
        dataset_name,
        split=split,
        as_supervised=True,
        with_info=True,
        # data_dir=os.path.join(data_dir, dataset_name),
        data_dir=data_dir,
        download=False,
    )

    ds = prepare_dataset(
        dataset,
        batch_size=batch_size,
        augment=augment,
        shuffle=split == "train",
    )

    if return_info:
        return ds, info
    return ds
