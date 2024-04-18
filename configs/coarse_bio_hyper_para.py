from easydict import EasyDict
hp = EasyDict({
    'model': 'Coarse',
    'data': {
        "train": 'Biofors_train',
        "sample_num": [1000],
        'test': 'Biofors_test',
        "img_size": 256,
    },
    "train": {
        "epochs": 200,
        "batch_size": 8,
    },
    "loss": {
        "seg": {
            "enable": True,
            "name": ["BCE", 'dice'],
            "weight": [0.7, 0.3],
            "size": 256,
            "pos_weight": [8.0, 1.0],
        },
        "cls": {
            "enable": True,
            "name": "BCE",
            "weight": 0.35,
            "pos_weight": 1.0,
        }
    },
    "optimizer": {
        "name": "Adam",
        "lr": 1e-4,
        "b1": 0.9,
        "b2": 0.999,
        "momentum": 0.9,
    }
})
