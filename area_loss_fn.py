import torch.nn.functional as F

def add_dimensions_loss_for_each_scale(dimensions_to_logits,
                               scales_to_logits,
                               labels,
                               num_classes,
                               upsample_logits=True): 
    