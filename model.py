import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions * frequencies
        encoding = torch.zeros(x.shape[1], self.d_model, device=x.device)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :self.d_model // 2].cos()
        return self.dropout(x + encoding.to(x.dtype).unsqueeze(0))


class IntraVideoTCCLoss(nn.Module):
    def __init__(self, temperature, pair_sampler=None):
        super().__init__()
        self.temperature = temperature
        self.pair_sampler = pair_sampler

    def forward(self, embeddings, padding_mask):
        if self.pair_sampler is None:
            raise NotImplementedError("The intra-video recurrence sampler is an internal component.")
        losses = []
        for sequence, masked in zip(embeddings, padding_mask):
            sequence = sequence[~masked]
            if len(sequence) < 2:
                continue
            anchors, positives, comparison_mask = self.pair_sampler(sequence.detach())
            if anchors.numel() == 0:
                continue
            normalized = F.normalize(sequence, dim=-1)
            logits = normalized[anchors] @ normalized.T / self.temperature
            rows = torch.arange(len(anchors), device=sequence.device)
            comparison_mask = comparison_mask.clone()
            comparison_mask[rows, positives] = True
            denominator = logits.masked_fill(~comparison_mask, -torch.inf).logsumexp(dim=-1)
            losses.append(denominator - logits[rows, positives])
        return torch.cat(losses).mean() if losses else embeddings.sum() * 0


def adaptive_loss_weight(
    class_loss, tcc_loss, epoch, base_alpha, warmup_epochs, beta, alpha_min, alpha_max
):
    warmup = min(epoch / warmup_epochs, 1.0)
    ratio = class_loss.detach() / (2 * (tcc_loss.detach() + 1e-8))
    adaptation = 1 + beta * torch.tanh(ratio)
    return (base_alpha * warmup * adaptation).clamp(alpha_min, alpha_max)


class TWCRAC_Transformer(nn.Module):
    def __init__(
        self,
        num_classes,
        *,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        embedding_dim,
        tcc_temperature,
        dim=99,
        dropout=0.1,
        pair_sampler=None,
    ):
        super().__init__()
        self.input_embedding = nn.Linear(dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.embedding_projection = nn.Linear(d_model, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes),
        )
        self.tcc_loss = IntraVideoTCCLoss(tcc_temperature, pair_sampler)

    @property
    def period_head(self):
        return self.classifier

    def forward(self, x, padding_mask=None, action_ids=None):
        single_video = x.ndim == 2
        if single_video:
            x = x.unsqueeze(0)
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(1, -1)
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        features = self.pos_encoder(self.input_embedding(x))
        features = self.transformer_encoder(features, src_key_padding_mask=padding_mask)
        logits = self.period_head(features)
        probabilities = logits.sigmoid()
        if action_ids is None:
            valid = (~padding_mask).unsqueeze(-1)
            video_scores = (probabilities * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            action_ids = video_scores.argmax(dim=-1)
        action_ids = torch.as_tensor(action_ids, device=x.device).long().reshape(-1)
        indices = action_ids[:, None, None].expand(-1, x.shape[1], 1)
        period_scores = probabilities.gather(-1, indices).squeeze(-1)
        output = {
            "logits": logits,
            "embeddings": self.embedding_projection(features),
            "period_scores": period_scores.masked_fill(padding_mask, 0),
        }
        return {key: value.squeeze(0) for key, value in output.items()} if single_video else output

    def compute_loss(self, poses, video_labels, epoch, weighting, padding_mask=None):
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        if padding_mask is None:
            padding_mask = torch.zeros(poses.shape[:2], dtype=torch.bool, device=poses.device)
        else:
            padding_mask = padding_mask.reshape(poses.shape[:2])
        output = self(poses, padding_mask)
        labels = video_labels.to(device=poses.device).long().reshape(-1)
        targets = F.one_hot(labels, num_classes=output["logits"].shape[-1])
        targets = targets.to(output["logits"].dtype).unsqueeze(1).expand_as(output["logits"])
        valid_frames = ~padding_mask
        class_loss = F.binary_cross_entropy_with_logits(
            output["logits"][valid_frames], targets[valid_frames]
        )
        tcc_loss = self.tcc_loss(output["embeddings"], padding_mask)
        alpha = adaptive_loss_weight(class_loss, tcc_loss, epoch, **weighting)
        return {
            "loss": (1 - alpha) * class_loss + alpha * tcc_loss,
            "class_loss": class_loss,
            "tcc_loss": tcc_loss,
            "alpha": alpha,
        }
