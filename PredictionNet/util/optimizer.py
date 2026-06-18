from torch import optim as optim
import itertools


def build_optimizer(args, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """

    if args.model_type == 'clip':
        optimizer = optim.AdamW([{'params': itertools.chain(model.parameters())}], eps=1e-6, betas=(0.9, 0.98), #0.98
                                lr=args.lr, weight_decay=args.weight_decay)
        
        # optimizer = optim.SGD([{'params': itertools.chain(model.parameters())}], lr=1e-7)
        
    #     # 获取除 model.clip.visual 以外的所有其他参数
    #     other_params = [param for name, param in model.named_parameters() if "clip.visual" not in name]
    #     optimizer = optim.AdamW([
    # {'params': model.clip.visual.parameters(), 'lr': 1e-7},
    # {'params': other_params, 'lr': 1e-5}], eps=1e-6, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    
    

    else:
        raise NotImplementedError(f"Unkown model: {args.model_type}")

    return optimizer


def set_weight_lr(args, model):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if
                       "backbone" in n and p.requires_grad and 'proj' not in n],
            "lr": args.lr_backbone,
        }
    ]
    return param_dicts
