from utils.loss_func import BinaryDiceLoss, MyBCELoss


def get_loss(pred_flat, true_flat, loss_dict, index=-1, uc_map=None):
    func_dict = {
        "BCE": MyBCELoss,
        "DICE": BinaryDiceLoss,
    }
    loss_value = 0.0
    if not isinstance(loss_dict.name, list):
        loss_dict.weight, loss_dict.pos_weight = [loss_dict.weight], [loss_dict.pos_weight]
        loss_dict.name = [loss_dict.name]
        loss_dict.label_smoothing_value = [loss_dict.label_smoothing_value]

    for i in range(len(loss_dict.name)):
        temp_true = true_flat.clone()
        temp_name = loss_dict.name[i].upper()
        loss_value = loss_value + loss_dict.weight[i] * func_dict[temp_name](
            pos_weight=loss_dict.pos_weight[i]
        )(pred_flat, temp_true)

    return loss_value
