from easydict import EasyDict
import time
import os.path as osp

# 配置
base_dir = '/home/hdd1/linxun/research/imd'
config_dict = EasyDict({
    # ----------------------------------------------------------------------------------------------------
    "logs_dir": osp.join(base_dir, f"logs/{str(time.time()).replace('.', '-')}"),  # 日志目录
    "dataset_dir": f"/home/hdd1/share/public_data/imd_2023",  # 所有数据集的根目录
    "bio_dataset_dir": f"/home/hdd1/share/public_data/bio_imd_2023",  # bio数据集的根目录
    "data_txt_dir": osp.join(base_dir, 'data'),
    "bio_data_txt_dir": osp.join(base_dir, 'data/bio'),
    #  ---------------------------------------------------------------------------------------------------
    "save_checkpoint": True,  # 是否每隔一段epoch保存一次
    "checkpoint_epochs": 30,  # 每隔多少epoch保存一次
    "eval_epochs": 1,  # 每隔多少epoch验证一次
    "save_best": True,  # 是否保存指标值最高的权重
    #  ---------------------------------------------------------------------------------------------------
    "fine_tuning": False,  # 是否fine tune
    "pkl_dir": "/home/hdd1/linxun/research/imd/logs/1684758948-8103878/checkpoint.pkl",  # 如果你需要fine-tune的话，在这里写上模型的路径
})
