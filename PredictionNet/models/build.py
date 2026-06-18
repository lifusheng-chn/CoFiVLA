from util.misc import load_pretrained
from .clip import Clip


def build_model(args, device):
    if args.model_type == 'clip':
        model, preprocess = Clip(args, device)
        return model, preprocess
    
    
    # if args.model_type == 'act':
    #     backbone = build_backbone(args) # 返回的网络可以提取backbone的各阶段特征及相对应的位置编码
    #     model_txt = HierAttNet(args.word_hidden_size, args.sent_hidden_size, args.train_batch_size,
    #                            args.word2vec_path, args.max_sent_length, args.max_word_length) # 返回文本特征
    #     model = ACT(backbone=backbone, model_txt=model_txt, d_model=args.d_model, nhead=args.nheads)


    else:
        raise NotImplementedError(f"Unkown model: {args.model_type}")

    return model


if __name__ == '__main__':
    import argparse
    import torch
    from util.misc import nested_tensor_from_tensor_list

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=256)
    config = parser.parse_args()
    model = build_model(config)
    model = model.to(devices)

    x = torch.randn(8, 3, 224, 224)
    x = nested_tensor_from_tensor_list(x)

    x = x.to(devices)
    out = model(x)
    print(out['pred_logits'].shape)
