import numpy as np


def assign_learning_rate(optimizer, new_lr):
   for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
       

# 在 warmup 阶段根据步数逐步增加学习率，直到达到基础学习率 base_lr
def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length



# 用于根据当前的训练步数调整学习率
# 优化器、基础学习率、预热步数、训练的总步数
def cosine_lr(optimizer, base_lr, warmup_length, steps):  # 判断当前的步数是否还处于 warmup 阶段（即小于 warmup_length）
    def _lr_adjuster(step):
        if step < warmup_length:  # 预热阶段，根据步数逐步增加学习率，直到达到基础学习率 base_lr
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:  #  当训练步数超过 warmup_length 后，进入余弦退火阶段。
            e = step - warmup_length
            es = steps - warmup_length
            # 通过余弦函数计算当前步数对应的学习率。这个公式实现了余弦退火：在训练开始时，学习率较高；随着训练的进行，学习率逐渐降低，最后收敛到一个较低的值。
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        # 负责将计算好的学习率 lr 分配给优化器 optimizer，使其在下一步训练时使用这个学习率。
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster
