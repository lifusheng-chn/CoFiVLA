from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy


from .model_longclip import CLIP
from .aesclip import tokenize


class AESMODEL(nn.Module):
    def __init__(self, clip_model: CLIP, device):
        super().__init__()
        self.feature_dim = 512
        self.hideen_dim = 768
        self.clip = clip_model
        self.device = device
        self.logit_scale = clip_model.logit_scale
        self.text_encoder =  TextEncoder(self.clip)

        self.quality_prompt_learner = QualityPromptLearner(dtype=torch.float32, token_embedding=clip_model.token_embedding, device=self.device)
        self.self_att = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8, batch_first=True)
        self.cross_att = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=1, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.ReLU(),
            # nn.Dropout(0.),
            nn.Linear(self.feature_dim * 4, self.feature_dim)
        )
        self.encoder_proj = nn.Linear(self.hideen_dim, self.feature_dim)
        nn.init.kaiming_normal_(self.encoder_proj.weight, a=0, mode='fan_out')
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

        self.img_ada1 = nn.Sequential(
            nn.Linear(self.hideen_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.hideen_dim),
        )
        self.img_ada2 = nn.Sequential(
            nn.Linear(self.hideen_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.hideen_dim),
        )
        self.img_ada3 = nn.Sequential(
            nn.Linear(self.hideen_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.hideen_dim),
        )
        self.img_ada4 = nn.Sequential(
            nn.Linear(self.hideen_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.hideen_dim),
        )
        # self.img_ada5 = nn.Sequential(
        #     nn.Linear(self.hideen_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.hideen_dim),
        # )
        # self.img_ada6 = nn.Sequential(
        #     nn.Linear(self.hideen_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.hideen_dim),
        # )


        
    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False
    def lock_text(self):
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.clip.visual.parameters():
            param.requires_grad = True
            


    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text):
        return self.clip.encode_text(text)



    def aes_fc(self, image_features):
        # output = image_features.flatten(1)
        # output = self.fc1(output)
        output = image_features
        output = self.fc(output)
        

        return output


    def forward(self, image
    ):
     
        batch_size = image.shape[0]
        image_feature, outputs = self.encode_image(image) if image is not None else None
        image_feature1 = self.img_ada1(outputs[0][: ,0, :])
        image_feature2 = self.img_ada2(outputs[1][: ,0, :])
        image_feature3 = self.img_ada3(outputs[2][: ,0, :])
        image_feature4 = self.img_ada4(outputs[3][: ,0, :])

        image_features = image_feature1 + image_feature2 + image_feature3 + image_feature4 
        image_features = self.encoder_proj(image_features)
        prompts, tokenized_featuress = self.quality_prompt_learner()
        prompt_features = self.text_encoder(prompts, tokenized_featuress)  # 形状: (6, 512)

        # query, _ = self.self_att(image_feature.unsqueeze(1), image_feature.unsqueeze(1), image_feature.unsqueeze(1))
        query =  image_feature.unsqueeze(1)
        key_value = prompt_features.unsqueeze(0).expand(batch_size, -1, -1)  # 形状为 (bs, 6, feature_dim)
        
        similarity_scores = torch.matmul(query, key_value.transpose(1, 2))  # 形状: (bs, 1, 6)
        probabilities = F.softmax(similarity_scores, dim=2)
        weighted_features = probabilities.unsqueeze(-1) * prompt_features.unsqueeze(0)  # (16, 1, 5, dim)
        cross_output = torch.sum(weighted_features, dim=2).squeeze(1)  # (16, 1, dim)
        cross_output = self.feed_forward(cross_output)  # 形状为 (bs, 1 feature_dim)
        

        combined_features = image_features + cross_output.squeeze(1)
        pre_score = self.aes_fc(combined_features)




        # return pre_score

        return {
            # "loss_itcl": loss_itcl,
            # "loss_itcs": loss_itcs,
            "image_features": image_feature,
            "logit_scale": self.logit_scale.exp(),
            "pre_score": pre_score,
            "probabilities": probabilities,
        }


class QualityPromptLearner(nn.Module):
    def __init__(self, dtype, token_embedding, device):
        super().__init__()
        # 定义质量评估级别
        qualitys = [
            'terrible', 'bad', 'average', 'good', 'perfect'
        ]
        # qualitys = [
        #     'terrible', 'bad',  'average', 'good', 'perfect','composition', 'lighting', 'shot', 'color', 'texture'
        # ]
        ctx_dim = 512  # 特征维度
        n_ctx = 8  # 上下文 token 的数量

        # 为每个质量级别初始化上下文向量
        print("初始化质量级别的特定上下文")
        ctx_vectors = torch.empty(len(qualitys), n_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors) # 可学习的参数

        # 初始化质量标签的 token 嵌入
        prompts = ["X " * n_ctx + q for q in qualitys]
        tokenized_feats = torch.cat([tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_feats).type(dtype)
            # embedding = embedding.to(device)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS token
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS token

        self.n_ctx = n_ctx
        self.tokenized_feats = tokenized_feats  # torch.Tensor
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [prefix, ctx, suffix],
                dim=1,
            )
        else:
            raise ValueError("Unsupported class token position")
        return prompts, self.tokenized_feats
    





    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_featuress):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_featuress.argmax(dim=-1)] @ self.text_projection
        return x
    
    


