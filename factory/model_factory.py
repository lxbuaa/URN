from SOTA.CATNET.hrnet import get_hrnet
from SOTA.EMT.emt import get_emt
from SOTA.GSR.gsr import get_gsr
from SOTA.IFOSN.full_net import get_IFOSN_build
from SOTA.MVSS.mvssnet import get_mvss
from SOTA.PraNet.PraNet_Res2Net import get_pranet
# from SOTA.UGNet.ugnet import get_ug_coarse
from SOTA.SegFormer import get_seg_former
# from SOTA.UGNetv2.coarse_net import get_coarse_net
# from SOTA.UGNetv2.fine_net import get_fine_net
from SOTA.BSANet.BSANet import get_bsa
from SOTA.PSCC.full_net import get_psccnet
from SOTA.TruFor.src.models.cmx.builder_np_conf import get_trufor
from UGNetv2 import get_coarse_net
from UGNetv2 import get_fine_net
from UGNetv2 import get_fine_um, get_fine_sa, get_fine_ca, get_fine_depth, get_fine_no_a, \
    get_fine_fully, get_fine_lug_no_d_no_w, get_fine_lug_d_no_w, get_fine_gug_d_no_w, get_fine_gug_d_w, get_fine_no_g, \
    get_fine_knn, get_fine_uema_u_d, get_fine_uema_d


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
