
def eval_prec_rec_f1_ir(preds, labels):
    true_map = 0
    mapped = 0
    m = len(preds)
    for i, pred_id in enumerate(preds):
        label = labels[i]  # str
        if pred_id is not None:  # str
            mapped += 1
            if pred_id == label:
                true_map += 1
    precision = true_map / mapped
    recall = true_map / m
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
