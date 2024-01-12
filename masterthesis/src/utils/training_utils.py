import numpy as np
import random

from ..data.cube_datasets import SlicedCubeDataset, TestWithZerosSlicedCubeDataset


def make_training_and_testing_data(
    train_test_split,
    train_test_seeds,
    stride=1,
    redshift=1.0,
    random_seed=42,
    transforms: bool = True,
):
    random.seed(random_seed)
    random.shuffle(train_test_seeds)

    array_length = len(train_test_seeds)
    assert (
        abs(sum(train_test_split) - 1.0) < 1e-6
    ), "Train and test split does not sum to 1."
    train_length = int(array_length * train_test_split[0])
    test_length = int(array_length * train_test_split[1])
    train_seeds = train_test_seeds[:train_length]
    test_seeds = train_test_seeds[train_length:]

    # Make datasets
    print("Making datasets...")
    print(f"Training set: {len(train_seeds)} seeds")
    train_dataset = SlicedCubeDataset(
        stride=stride,
        redshift=redshift,
        seeds=train_seeds,
        use_transformations=transforms,
    )
    print(f"Test set: {len(test_seeds)} seeds")
    test_dataset = SlicedCubeDataset(
        stride=stride,
        redshift=redshift,
        seeds=test_seeds,
        use_transformations=transforms,
    )
    return train_dataset, test_dataset


def make_test_with_zeros_training_and_testing_data(
    train_test_split,
    train_test_seeds,
    stride=1,
    redshift=1.0,
    random_seed=42,
    transforms: bool = True,
):
    random.seed(random_seed)
    random.shuffle(train_test_seeds)

    array_length = len(train_test_seeds)
    assert (
        abs(sum(train_test_split) - 1.0) < 1e-6
    ), "Train and test split does not sum to 1."
    train_length = int(array_length * train_test_split[0])
    test_length = int(array_length * train_test_split[1])
    train_seeds = train_test_seeds[:train_length]
    test_seeds = train_test_seeds[train_length:]

    # Make datasets
    print("Making datasets...")
    print(f"Training set: {len(train_seeds)} seeds")
    train_dataset = TestWithZerosSlicedCubeDataset(
        stride=stride,
        redshift=redshift,
        seeds=train_seeds,
        use_transformations=transforms,
    )
    print(f"Test set: {len(test_seeds)} seeds")
    test_dataset = TestWithZerosSlicedCubeDataset(
        stride=stride,
        redshift=redshift,
        seeds=test_seeds,
        use_transformations=transforms,
    )
    return train_dataset, test_dataset
