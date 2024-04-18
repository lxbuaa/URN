from torch.utils.data import DataLoader
import os.path as osp
from data.my_bio_dataset import BioBaseDataset, BioDataset


def get_data(hyper_para, config, split='train'):
    """
    Get dataloader
    :param hyper_para: hyperparameters
    :param config: config files
    :param split: train or test
    :return: dataloader
    """
    print(config)
    dataset_dict = {
        "SciSp_C_pristine_train": "SciSp_C_pristine_train",
        "SciSp_C_pristine_test": "SciSp_C_pristine_test",
        "SciSp_C_splicing_train": "SciSp_C_splicing_train",
        "SciSp_C_splicing_test": "SciSp_C_splicing_test",

        "SciSp_H_train": "SciSp_H_train",
        "SciSp_H_test": "SciSp_H_test",

        "Biofors_pristine_train": "Biofors_pristine_train",
        "Biofors_pristine_test": "Biofors_pristine_test",
        "Biofors_splicing_train": "Biofors_splicing_train",
        "Biofors_splicing_test": "Biofors_splicing_test",

        "RSIIL_pristine_train": "RSIIL_pristine_train",
        "RSIIL_pristine_test": "RSIIL_pristine_test",
        "RSIIL_splicing_train": "RSIIL_splicing_train",
        "RSIIL_splicing_test": "RSIIL_splicing_test",
    }
    dataset_name = hyper_para.data[split]
    if split == 'train':
        dataset = get_dataset_by_name(config, hyper_para, dataset_dict[dataset_name], split='train')
        batch_size = hyper_para.train.batch_size
        data_loader = get_dataloader(dataset, 'train', batch_size)
    elif split == 'test':
        dataset = get_dataset_by_name(config, hyper_para, dataset_dict[dataset_name], split='test')
        data_loader = get_dataloader(dataset, 'test')

    print(f'Load {dataset_name}, about {len(dataset)} images')

    return data_loader


def get_dataloader(dataset: BioBaseDataset, split: str, batch_size=32):
    """
    :param dataset: dataset in pytorch format
    :param split: train or test
    :param batch_size: batch size for input data
    :return: dataloader
    """
    data_loader = DataLoader(
        dataset,
        num_workers=2 if split == 'train' else 1,
        pin_memory=True if split == 'train' else False,
        shuffle=True if split == 'train' else False,
        batch_size=batch_size if split == 'train' else 1
    )
    return data_loader


def get_dataset_by_name(config, hp, dataset_name, sample_num=-1, split='train', post_process=None):
    txt_name = f'{dataset_name}.txt'
    return BioDataset(config, osp.join(config.bio_data_txt_dir, txt_name),
                      hp.data.img_size, split=split, sample_num=sample_num,
                      post_process=post_process)
