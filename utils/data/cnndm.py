from datasets import load_dataset, IterableDataset


def load_cnndm(args=None):
    assert args is not None, "args must be provided to load_cnndm"
    assert (
        args.dataset_config_name is not None
    ), "args.dataset_config_name must be None to load_cnndm"
    assert args.dataset_split is not None, "args.dataset_split must be None to load_cnndm"

    # load the regular dataset
    raw_dataset = load_dataset('cnn_dailymail', '3.0.0', split = 'train')

    def cnndm_generator():
        # the generator loop
        for ex in raw_dataset:
            yield ex

    dataset = IterableDataset.from_generator(cnndm_generator)
    return dataset
