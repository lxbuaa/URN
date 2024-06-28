from UGNetv2 import get_coarse_net
from UGNetv2 import get_fine_net


def get_model(hyper_para):
    """
    Get network
    :param hyper_para: hyperparameters
    :return: Network
    """
    model_name = hyper_para.model
    model_dict = {
        "Coarse": get_coarse_net,
        "Fine": get_fine_net,
    }
    if model_name.find('Fine') == -1:
        net_model = model_dict[model_name]()
    else:
        net_model = model_dict[model_name]("Weight path of Coarse Net")

    return net_model
