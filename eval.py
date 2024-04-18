import torch
import tqdm
from utils.metric_collector import MetricCollector


def eval_after_train(network, data_loader, hyper_p, k=0.5):
    """
    Calculate metric values
    :param network: Network to use
    :param data_loader: Dataloader of test dataset
    :param hyper_p: Hyper parameters
    :param k: Threshold for calculating the metrics
    :return: Pixel-level and image-level metrics
    """
    metric_values = MetricCollector()
    # Stop gradients
    network.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(data_loader)):
            # Feed a batch to the network
            net_out = network(batch['img'].cuda())
            # Update pixel-level metrics
            if hyper_p.loss.seg.enable:
                main_pred_flat = torch.sigmoid(net_out['seg']).float().view(-1)
                main_true_flat = batch['mask'].cuda().float().view(-1)
                metric_values.update(main_pred_flat, main_true_flat, data_loader.batch_size, 'seg', k)

            # Update image-level metrics
            if hyper_p.loss.cls.enable:
                cls_pred_flat = torch.sigmoid(net_out['cls']).float().view(-1)
                cls_true_flat = batch['cls'].cuda().float().view(-1)
                metric_values.update(cls_pred_flat, cls_true_flat, data_loader.batch_size, 'cls', k)

    # Output metric values
    seg_res = metric_values.show('seg')
    print(metric_values.metrics.seg.f1_seg.avg, metric_values.metrics.seg.mcc_seg.avg)

    cls_res = metric_values.show('cls')
    print(metric_values.metrics.cls.auc_cls.avg, metric_values.metrics.cls.acc_cls.avg)
    print('\n')

    return seg_res, cls_res



