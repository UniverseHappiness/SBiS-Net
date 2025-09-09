from __future__ import division, print_function
import random

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torchvision.models as models
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pad_sequence

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(
        self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True
    ):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2
            )
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MiniUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0, bilinear=True):
        super(MiniUpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.bilinear = self.params["bilinear"]
        self.dropout = self.params["dropout"]
        assert len(self.ft_chns) == 5
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.bilinear = self.params["bilinear"]
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0
        )
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0
        )
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0
        )
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0
        )

        self.out_conv = nn.Conv2d(
            self.ft_chns[0], self.n_class, kernel_size=3, padding=1
        )

    def forward(self, feature, is_contrastive=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        if is_contrastive:
        # 解码过程，逐步恢复空间分辨率
            d3 = self.up1(x4, x3)  # 第一个上采样层输出
            d2 = self.up2(d3, x2)  # 第二个上采样层输出
            d1 = self.up3(d2, x1)  # 第三个上采样层输出
            d0 = self.up4(d1, x0)  # 第四个上采样层输出
            return [d0, d1, d2, d3]
        

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output
    
# class SBGAD(nn.Module):
#     def __init__(self, in_channels, attr_dim=16, p=0.1, k=2):
#         super(SBGAD, self).__init__()
#         self.p = p  # percentile for boundary region
#         self.k = k  # number of clusters
#         self.attr_dim = attr_dim

#         # Projection and reconstruction MLP
#         self.projection = nn.Sequential(
#             nn.Conv2d(in_channels, 2 * attr_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2 * attr_dim, attr_dim, kernel_size=1)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(attr_dim, 2 * attr_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(2 * attr_dim, attr_dim)
#         )
        
#     def compute_boundary_map(self, P):
#         """Gradient-based soft boundary map"""
#         grad_x = torch.abs(P[..., :, 1:] - P[..., :, :-1])
#         grad_y = torch.abs(P[..., 1:, :] - P[..., :-1, :])
#         grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
#         grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
#         boundary = grad_x + grad_y  # [B,K,H,W]
#         boundary_map = boundary.sum(dim=1, keepdim=True)  # [B,1,H,W]
#         return (boundary_map - boundary_map.min()) / (boundary_map.max() - boundary_map.min() + 1e-6)

#     def threshold_top_percentile(self, B, p):
#         """Get binary mask for top-p percentile values"""
#         B_flat = B.view(B.size(0), -1)
#         k = int(B_flat.size(1) * p)
#         thresholds = torch.kthvalue(B_flat, B_flat.size(1) - k, dim=1).values
#         R = (B >= thresholds.view(-1, 1, 1, 1)).float()
#         return R

#     def forward(self, Fe, P):
#         B, C, H, W = Fe.shape

#         # Step 1: Boundary region mask
#         B_map = self.compute_boundary_map(Fe)
#         R = self.threshold_top_percentile(B_map, self.p)  # [B,1,H,W]

#         # Step 2: Attribute projection
#         A = self.projection(Fe)  # [B,D,H,W]

#         # Step 3: Region clustering
#         A_flat = A.permute(0, 2, 3, 1).reshape(-1, A.size(1))  # [B*H*W, D]
#         R_flat = R.view(-1)  # [B*H*W]
#         boundary_features = A_flat[R_flat > 0]  # [N, D]

#         # Step 4: Reconstruction loss
#         loss_rec = 0.0
#         if boundary_features.size(0) > 0:
#             N = boundary_features.size(0)
#             idx = torch.randperm(N)
#             half = N // 2
#             A1 = boundary_features[idx[:half]]  # [half, D]
#             A2 = boundary_features[idx[half:]]  # [half, D]

#             # Compute A1_pred and its loss
#             A1_pred = self.decoder(A2.detach())  # Reconstruct A1 from A2
#             loss_rec = F.l1_loss(A1_pred, A1, reduction='mean')
#         else:
#             loss_rec = torch.tensor(0.0, device=Fe.device)
#         return loss_rec

class LocalPositionalEncoding(nn.Module):
    def __init__(self, k=5, embed_dim=64, use_learnable=True):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.use_learnable = use_learnable

        # 生成相对坐标偏移矩阵 (k*k, 2)
        grid = torch.stack(torch.meshgrid(
            torch.arange(-(k // 2), k // 2 + 1),
            torch.arange(-(k // 2), k // 2 + 1),
            indexing='ij'
        ), dim=-1).view(-1, 2)  # shape: [k*k, 2]

        if use_learnable:
            # 可学习的嵌入表：将k*k种偏移映射到embed_dim维向量
            self.embed = nn.Embedding(k * k, embed_dim)
            # 为每个偏移生成唯一索引 (0~k*k-1)
            indices = (grid[:, 0] + k // 2) * k + (grid[:, 1] + k // 2)
            self.register_buffer('indices', indices.long())
        else:
            # 正弦编码：预计算所有偏移的位置编码
            pe = torch.zeros(grid.size(0), embed_dim)
            for i, (dx, dy) in enumerate(grid):
                pe_x = self._sinusoid_encoding(dx, embed_dim // 2)
                pe_y = self._sinusoid_encoding(dy, embed_dim // 2)
                pe[i] = torch.cat([pe_x, pe_y], dim=0)
            self.register_buffer('pe', pe)  # 保存所有邻域编码，不平均

    def _sinusoid_encoding(self, delta, dim):
        """为单个偏移量生成正弦编码"""
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() *
            (-math.log(10000.0) / dim)
        ).to(delta.device)
        encoding = torch.zeros(dim, device=delta.device)
        encoding[0::2] = torch.sin(delta * div_term)
        encoding[1::2] = torch.cos(delta * div_term)
        return encoding

    def forward(self, x):
        """
        输入x: [B, C, H, W]
        输出: [B, C+embed_dim, H, W]
        """
        B, _, H, W = x.shape

        # 生成绝对位置编码（二维正弦编码）
        grid_i, grid_j = torch.meshgrid(torch.arange(H, device=x.device),
                                        torch.arange(W, device=x.device), indexing='ij')
        grid_i, grid_j = grid_i.float(), grid_j.float()

        # 计算i和j的正弦编码
        div_term = torch.exp(
            torch.arange(0, self.embed_dim // 2, 2).float() *
            (-math.log(10000.0) / (self.embed_dim // 2))
        ).to(x.device)

        # 编码i的坐标
        i_encoding = grid_i.unsqueeze(-1) * div_term
        pe_i = torch.stack([torch.sin(i_encoding), torch.cos(i_encoding)], dim=-1)
        pe_i = pe_i.view(H, W, -1)  # [H, W, embed_dim//2]

        # 编码j的坐标
        j_encoding = grid_j.unsqueeze(-1) * div_term
        pe_j = torch.stack([torch.sin(j_encoding), torch.cos(j_encoding)], dim=-1)
        pe_j = pe_j.view(H, W, -1)  # [H, W, embed_dim//2]

        abs_pe = torch.cat([pe_i, pe_j], dim=-1)  # [H, W, embed_dim]

        # 获取邻域位置编码
        if self.use_learnable:
            embeddings = self.embed(self.indices)  # [k*k, embed_dim]
        else:
            embeddings = self.pe.to(x.device)  # [k*k, embed_dim]

        # 将绝对编码与邻域编码相加后平均
        combined = abs_pe.unsqueeze(2) + embeddings.view(1, 1, -1, self.embed_dim)
        lpe = combined.mean(dim=2)  # [H, W, embed_dim]

        # 调整维度并扩展至批次
        lpe = lpe.permute(2, 0, 1).unsqueeze(0).expand(B, -1, H, W)

        # 拼接原特征与位置编码
        return torch.cat([x, lpe], dim=1)

class LPEReconstructionFromDeep(nn.Module):
    def __init__(self, in_channels=256, embed_dim=64, out_channels=1, use_learnable=True, k=5):
        super().__init__()
        self.lpe = LocalPositionalEncoding(k=k, embed_dim=embed_dim, use_learnable=use_learnable)

        # LPE 会输出 in_channels + embed_dim

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels + embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MiniUpBlock(256, 128),
            MiniUpBlock(128, 64),
            MiniUpBlock(64, 32),
            MiniUpBlock(32, 16),

            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(in_channels + embed_dim, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16→32

        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32→64

        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64→128

        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128→256

        #     nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        #     nn.Sigmoid()  # 输出像素值 0~1
        # )

    def forward(self, x):
        x = self.lpe(x)  # 添加位置编码
        return self.decoder(x)


class SBGAD(nn.Module):
    def __init__(self, in_channels, attr_dim=16, p=0.1, k=2):
        super(SBGAD, self).__init__()
        self.p = p  # percentile for boundary region
        self.k = k  # number of clusters
        self.attr_dim = attr_dim

        # Projection and reconstruction MLP
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 2 * attr_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * attr_dim, attr_dim, kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(attr_dim, 2 * attr_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * attr_dim, attr_dim)
        )

    def compute_boundary_map(self, P):
        """Gradient-based soft boundary map"""
        grad_x = torch.abs(P[..., :, 1:] - P[..., :, :-1])
        grad_y = torch.abs(P[..., 1:, :] - P[..., :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        boundary = grad_x + grad_y  # [B,K,H,W]
        boundary_map = boundary.sum(dim=1, keepdim=True)  # [B,1,H,W]
        return (boundary_map - boundary_map.min()) / (boundary_map.max() - boundary_map.min() + 1e-6)

    def threshold_top_percentile(self, B, p):
        """Get binary mask for top-p percentile values"""
        B_flat = B.view(B.size(0), -1) # torch.Size([12, 65536])
        k = int(B_flat.size(1) * p) # 前10%元素
        thresholds = torch.kthvalue(B_flat, B_flat.size(1) - k, dim=1).values # torch.Size([12])
        R = (B >= thresholds.view(-1, 1, 1, 1)).float()
        return R
    
    def fast_clustering(self, A, R):
        """
        Fast approximate clustering using random sampling and pairwise distances
        A: [B,D,H,W], R: [B,1,H,W]
        Returns: (A1, A2) attribute samples in cluster-1 and cluster-2 (padded tensors)
        """
        if R.size(2) != A.size(2) or R.size(3) != A.size(3):
            R = F.interpolate(R, size=(A.size(2), A.size(3)), mode='nearest')
        
        B, D, H, W = A.shape
        A_flat = A.permute(0, 2, 3, 1).reshape(B, H*W, D)  # [B, H*W, D]
        R_flat = R.view(B, H*W)  # [B, H*W]
        
        # Get masked attributes for each batch
        masked_A = [A_flat[b][R_flat[b].bool()] for b in range(B)]
        
        # Filter batches with enough samples
        valid_batches = [b for b in range(B) if masked_A[b].shape[0] >= self.k]
        if not valid_batches:
            return None, None
        
        # Random sampling for cluster centers (much faster than full k-means)
        sampled_A = torch.cat([masked_A[b] for b in valid_batches])
        rand_idx = torch.randperm(sampled_A.shape[0])[:self.k]
        centers = sampled_A[rand_idx]  # [k, D]
        
        # Assign clusters based on nearest center
        A1_list, A2_list = [], []
        for b in valid_batches:
            dists = torch.cdist(masked_A[b], centers)  # [N, k]
            labels = torch.argmin(dists, dim=1)
            A1_list.append(masked_A[b][labels == 0])
            A2_list.append(masked_A[b][labels == 1])
        
        # Pad sequences to handle variable lengths
        A1_padded = pad_sequence(A1_list, batch_first=True) if A1_list else None
        A2_padded = pad_sequence(A2_list, batch_first=True) if A2_list else None
        
        return A1_padded, A2_padded
    def region_clustering(self, A, R, max_components=100):
        """
        Clustering attributes within each soft boundary region.
        A: [B,D,H,W], R: [B,1,H,W]
        Return: (A1, A2) attribute samples in cluster-1 and cluster-2 (list of tensors)
        """
        if R.size(2) != A.size(2) or R.size(3) != A.size(3):
            R = F.interpolate(R, size=(A.size(2), A.size(3)), mode='nearest')
        B, D, H, W = A.shape
        A1, A2 = [], []

        for b in range(B):
            mask = R[b, 0]
            if mask.sum() < 10:
                continue
            coords = mask.nonzero(as_tuple=False)  # [N, 2]
            attrs_raw = A[b, :, coords[:, 0], coords[:, 1]].permute(1, 0)  # [N,D]
            if attrs_raw .size(0) < self.k:
                continue
            attrs_np = attrs_raw.detach().cpu().numpy()  # 仅用于聚类

            kmeans = KMeans(n_clusters=self.k, n_init=5)
            labels = kmeans.fit_predict(attrs_np)
            labels = torch.from_numpy(labels).to(A.device)
            # [N, D]
            # (1608,16)


            # labels = kmeans.fit_predict(attrs.detach().cpu().numpy())
            # labels = torch.from_numpy(labels).to(A.device)

            A1.append(attrs_raw[labels == 0])
            A2.append(attrs_raw[labels == 1])


        return A1, A2

    def forward(self, Fe, P):
        """
        F: [B,C,H,W] - decoder feature map
        P: [B,K,H,W] - predicted softmax probability map
        Returns: loss_sbgad (scalar)
        """
        # Step 1: Boundary region mask
        B_map = self.compute_boundary_map(Fe)  # [B,1,H,W]
        R = self.threshold_top_percentile(B_map, self.p)  # [B,1,H,W]

        # Step 2: Attribute projection
        A = self.projection(Fe)  # [B,D,H,W]

        # Step 3: Fast region clustering
        A1, A2 = self.fast_clustering(A, R)

        # Step 4: Cross reconstruction
        if A1 is None or A2 is None:
            return torch.tensor(0.0, device=Fe.device)
        
        # Ensure equal samples by taking minimum length
        min_samples = min(A1.size(1), A2.size(1))
        A1 = A1[:, :min_samples]
        A2 = A2[:, :min_samples]
        
        # Vectorized reconstruction
        A1_pred = self.decoder(A2)
        A2_pred = self.decoder(A1)
        
        loss_rec = F.l1_loss(A1_pred, A1, reduction='mean') + F.l1_loss(A2_pred, A2, reduction='mean')
        return loss_rec

class SCE_LRD(nn.Module):
    def __init__(self, alpha=5.0, beta=0.6, delta=0.6, tau=0.5, gamma=0.5, window_size=5, momentum=0.999):
        super(SCE_LRD, self).__init__()
        self.alpha = alpha      # sigmoid sharpness
        self.beta = beta        # activation sampling threshold
        self.delta = delta      # binary mask threshold (relative to max)
        self.tau = tau          # local density threshold
        self.gamma = gamma      # balance between activation and density
        self.window_size = window_size
        self.momentum = momentum
        self.register_buffer('S', None)  # similarity matrix

    def forward(self, F):
        """
        F: Tensor [B, C, H, W] — feature maps from encoder
        Return:
            F_perturbed: Tensor [B, C, H, W] — after structure-aware channel exchange
        """
        B, C, H, W = F.shape

        # ---------- Step 1: Compute channel similarity matrix ----------
        if self.S is None:
            self.S = torch.eye(C, device=F.device)
        with torch.no_grad():  # 新增
            # Normalize features across spatial dimensions
            F_flat = F.view(B, C, -1)
            F_norm = F_flat / (F_flat.norm(dim=2, keepdim=True) + 1e-6)  # [B, C, HW]
            S_batch = torch.bmm(F_norm, F_norm.transpose(1, 2))  # [B, C, C]
            S_batch = S_batch.mean(dim=0)  # average across batch
            self.S = self.momentum * self.S + (1 - self.momentum) * S_batch  # EMA update
        # print(self.S.shape)
        # ---------- Step 2: Generate perturbation mask using LRD sampling ----------
        M = self.local_response_density_sampling(F)
        # print(M.shape)
        # ---------- Step 3: Structure-aware channel exchange ----------

        # print(M.shape) # (24, C)
        permutation = torch.arange(C, device=F.device).unsqueeze(0).expand(B, -1).clone()

        for b in range(B):
            selected = torch.where(M[b] > 0.5)[0]
            unselected = torch.where(M[b] <= 0.5)[0]
            
            if len(selected) == 0 or len(unselected) == 0:
                continue

            # 批量计算所有selected通道对应的最相似unselected通道
            
            similarities = self.S[selected[:, None], unselected[None, :]]  # [S, U]
            # print(similarities)
            j_indices = similarities.argmax(dim=1)          # [S]
            j = unselected[j_indices]                       # [S]

            
            # 创建当前样本的置换副本
            perm = permutation[b].clone()
            
            # 执行交换：selected <-> j
            perm[selected] = j      # 将selected位置指向j
            perm[j] = selected      # 将j位置指回selected
            
            permutation[b] = perm

        # 应用批量置换 [B, C, H, W]
        F = F.gather(1, permutation.view(B, C, 1, 1).expand(-1, -1, H, W))
        
        return F
        # F_copy = F.clone()  # 避免原始张量被修改
        
        # for b in range(B):
        #     # print(M[b] > 0.5)
        #     selected = torch.where(M[b] > 0.5)[0]
        #     unselected = torch.where(M[b] <= 0.5)[0]

        #     for k in selected:
        #         # Find most similar unselected channel
        #         if len(unselected) == 0: continue
        #         j = unselected[torch.argmax(self.S[k, unselected])]
        #         # F[b, [k, j]] = F[b, [j, k]]  # exchange
        #         temp = F_copy[b, k].clone()
        #         F_copy[b, k] = F_copy[b, j]
        #         F_copy[b, j] = temp

        # return F_copy

    def local_response_density_sampling(self, F):
        """
        Implements LRD Sampling as described
        Input: F [B, C, H, W]
        Output: M [B, C] binary sampling mask
        """
        B, C, H, W = F.shape

        # 1. Global average activation a_i

        a = F.mean(dim=(2, 3))  # [B, C]
        a = torch.sigmoid(a)
        # print(a)
        # 2. Binary masks (delta thresholding)
        max_vals = F.view(B, C, -1).amax(dim=-1).view(B, C, 1, 1)
        # print(max_vals)
        M_bin = (F >= self.delta * max_vals).float()

        # 3. Local density map
        pool = nn.AvgPool2d(kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        density_map = pool(M_bin)  # [B, C, H, W]

        # 4. Compute average density per channel
        high_density = (density_map >= self.tau).float()
        d = high_density.mean(dim=(2, 3))  # [B, C]
   
        # 5. Combine into final score
        s = a**self.gamma * d**(1 - self.gamma)  # [B, C]
    
        s_max = s.max(dim=1, keepdim=True)[0]
 
        prob = torch.sigmoid(self.alpha * (s - self.beta * s_max))  # [B, C]

        # print(prob)
        # 6. Bernoulli sampling
        M = torch.bernoulli(prob)  # [B, C]
        return M


class BiDSA_UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(BiDSA_UNet, self).__init__()

        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu'
        }

      
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        # BiDSA 部分：定义卷积层
        self.conv1x1_1 = nn.Conv2d(256, 16, kernel_size=1)  # 用于语义注入
        self.conv3x3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # 用于门控
        self.conv3x3_2 = nn.Conv2d(16 + 256, 16, kernel_size=3, padding=1) # 新增
        self.conv1x1_2 = nn.Conv2d(16, 256, kernel_size=1)  # 用于结构增强
        self.sce_list = nn.ModuleList([
            SCE_LRD(),
            SCE_LRD(),
            SCE_LRD(),  # 用于 feature[2]
            SCE_LRD(),  # 用于 feature[3]
            SCE_LRD(),  # 用于 feature[4]
        ])

        self.reconstruction_decoder = LPEReconstructionFromDeep()

        # self.sbgad_list = nn.ModuleList([
        #     SBGAD(in_channels=16),
        #     # SBGAD(in_channels=32),
        #     # SBGAD(in_channels=64),
        #     # SBGAD(in_channels=128),
        #     # SBGAD(in_channels=256),
        # ])
        
        
    def semantic_injection(self, f_s, f_d):
        # 语义注入：深层特征注入到浅层特征中
        # print(f_s.shape) # torch.Size([12, 16, 256, 256])
        # print(f_d.shape) # torch.Size([12, 256, 16, 16])
        
        
        f_d_up = F.interpolate(f_d, size=f_s.shape[2:], mode='bilinear', align_corners=False) 
        
        # print(f_d_up.shape)
        f_d_proj =  self.conv1x1_1(f_d_up) 
        # f_concat = torch.cat([f_s, f_d_up], dim=1)  # [B, 16+256=272, H, W]
        # gate = torch.sigmoid(self.conv3x3_2(f_concat))  # [B, 16, H, W]
        # print(f_d_proj.shape)
        gate = torch.sigmoid(self.conv3x3(f_s))  
        
        return f_s + gate * f_d_proj

    def structure_reinforcement(self, f_s, f_d):
        # 结构增强：浅层特征注入到深层特征中
        # print(f_s.shape) # torch.Size([12, 16, 256, 256])
        # print(f_d.shape) # torch.Size([12, 256, 16, 16])
        f_s_down =F.max_pool2d(f_s, kernel_size=16, stride=16) 
        # print(f_s_down.shape)
        f_s_proj = self.conv1x1_2(f_s_down) 
        # print(f_s_proj.shape)
        return f_d + f_s_proj
    
    def forward(self, x, need_fp=False, need_bidsa=False, need_fp_sce=False, need_sbgad=False, need_rec=False):
        feature = self.encoder(x)

        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)
        if need_fp_sce:
            feature_aug = [f.clone() for f in feature]  # 克隆防止改动原始特征
            for i in range(len(feature)):
                feature_aug[i] = self.sce_list[i](feature[i])
            # out = self.decoder(feature)
            # out_aug = self.decoder(feature_aug)

            feature_cat = [torch.cat([f, f_aug], dim=0) for f, f_aug in zip(feature, feature_aug)]
            out_cat = self.decoder(feature_cat)
            out, out_aug = out_cat.chunk(2)

            return out, out_aug

        if need_sbgad:
            # outs = self.decoder(feature).detach()
            # outs_softmax = outs.softmax(dim=1)
            sbgad = SBGAD(in_channels=16).cuda()
            loss_rec = sbgad(feature[0], None)
            return loss_rec

        if need_bidsa:
            f_s = feature[0]  # 浅层特征
            f_d = feature[4]  # 深层特征
            f_s_injected = self.semantic_injection(f_s, f_d)
            f_d_reinforced = self.structure_reinforcement(f_s, f_d)
            # print(f_s_injected.shape)
            # print(f_d_reinforced.shape)
            # 进入解码器进行空间恢复
            if need_fp:
                return f_s_injected, f_d_reinforced, self.decoder([f_s_injected, f_d_reinforced, *feature[1:]])
            return f_s_injected, f_d_reinforced
        if need_rec:
             
            rec_output = self.reconstruction_decoder(feature[4])
            
            return rec_output

        # 普通的解码输出
        output = self.decoder(feature)
        return output

# class Sine(nn.Module):
#     """SIREN sine activation function"""
#     def __init__(self, w0=30):
#         super().__init__()
#         self.w0 = w0

#     def forward(self, x):
#         return torch.sin(self.w0 * x)


# class MLPRefiner(nn.Module):
#     """MLP 作为高分辨率特征优化模块"""
#     def __init__(self, in_features, hidden_features=64, out_features=None, depth=3):
#         super().__init__()
#         out_features = out_features or in_features
#         layers = []
#         for i in range(depth):
#             layers.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
#             layers.append(Sine())  # SIREN 激活
#         layers.append(nn.Linear(hidden_features, out_features))
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # 变换维度
#         x = self.mlp(x)
#         x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # 恢复形状
#         return x


# class UpBlock(nn.Module):
#     """Upsampling followed by ConvBlock or MLP"""
#     def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True, use_mlp=False):
#         super(UpBlock, self).__init__()
#         self.use_mlp = use_mlp
#         self.bilinear = bilinear

#         if use_mlp:  
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#             self.mlp = MLPRefiner(in_channels2 * 2, hidden_features=out_channels)
#         else:
#             if bilinear:
#                 self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
#                 self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#             else:
#                 self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
#             self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

#     def forward(self, x1, x2):
#         if self.bilinear and not self.use_mlp:
#             x1 = self.conv1x1(x1)
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         if self.use_mlp:
#             x = self.mlp(x)
#         else:
#             x = self.conv(x)
#         return x


# class Decoder(nn.Module):
#     def __init__(self, params):
#         super(Decoder, self).__init__()
#         self.params = params
#         self.ft_chns = self.params["feature_chns"]
#         self.n_class = self.params["class_num"]
#         self.bilinear = self.params["bilinear"]

#         self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear)
#         self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear)
#         self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear, use_mlp=True)
#         self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear, use_mlp=True)

#         self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

#     def forward(self, feature):
#         x0, x1, x2, x3, x4 = feature
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)  # 高分辨率，MLP 处理
#         x = self.up4(x, x0)  # 高分辨率，MLP 处理
#         output = self.out_conv(x)
#         return output



class PertDropout(nn.Module):
    def __init__(self, p=0.5):
        super(PertDropout, self).__init__()
        self.p = p
        self.dropouts = [
            nn.Dropout2d(p * 0.5).cuda(),  # Weak
            nn.Dropout2d(p * 1.5).cuda(),  # Strong
        ]
        self.len = len(self.dropouts)

    def __len__(self):
        return self.len

    def forward(self, x):
        rst = []
        for pert_dropout in self.dropouts:
            single_type = []
            for i, feat in enumerate(x):
                perted = pert_dropout(feat)
                single_type.append(perted)
            rst.append(single_type)
        return rst

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FAT(nn.Module):
    def __init__(self, num_channels, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
        super(FAT, self).__init__()
        self.num_channels = num_channels
        self.scale_range = scale_range
        self.shift_range = shift_range

    def forward(self, x):
        B, C, H, W = x.shape  # 获取特征的批量大小、通道数、高度和宽度
        
        # 生成随机缩放和偏移参数
        gamma = torch.empty((B, C, 1, 1), device=x.device).uniform_(*self.scale_range)  # [B, C, 1, 1]
        beta = torch.empty((B, C, 1, 1), device=x.device).uniform_(*self.shift_range)  # [B, C, 1, 1]
        
        return gamma * x + beta  # 进行仿射变换


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30):
        super(SIRENLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0  # 频率因子，控制隐式表示的高频能力
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.w0, 1 / self.w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))
    
# feature_per
class UNet_feature(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_feature, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        
        # 仅在高层特征上使用 SIREN
        # self.siren = SIRENLayer(256, 256)
        # self.siren = SIREN(in_features=256, hidden_features=256, hidden_layers=3, out_features=256)

        # self.se_blocks = nn.ModuleList([SEBlock(channels) for channels in [16, 32, 64, 128, 256]])  # 事先定义
        # self.fat_blocks = nn.ModuleList([FAT(channels) for channels in [16, 32, 64, 128, 256]])  # 事先定义
    # 修改UNet_unimatch的forward方法
    def forward(self, x, need_fp=False):
        feature = self.encoder(x)
        
        if need_fp:
            # 对高层特征添加SIREN扰动
            # feature[-1] = self.siren(feature[-1].flatten(2).transpose(1,2)).transpose(1,2).reshape_as(feature[-1])
            
            # 对每个特征层添加不同类型的噪声
            perturbed_features = []
            for idx, feat in enumerate(feature):
                if idx < 2:  # 低层特征使用高斯噪声
                    noise = torch.randn_like(feat) * 0.1
                    perturbed = feat + noise
                else:        # 高层特征使用dropout+通道噪声
                    perturbed = feat * torch.rand(feat.size(0), 1, 1, 1, device=feat.device) * 0.5
                    perturbed = nn.Dropout2d(0.3)(perturbed)
                perturbed_features.append(torch.cat([feat, perturbed], dim=1))
            
            outs = self.decoder(perturbed_features)
            return outs.chunk(2)
        else:
            return self.decoder(feature)

class UNet_contrastive(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_contrastive, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        
        # # 添加多尺度输出层
        # self.scale_heads = nn.ModuleList([
        #     nn.Conv2d(params['feature_chns'][3], class_num, 1),  # 128 通道 -> 32x32
        #     nn.Conv2d(params['feature_chns'][2], class_num, 1),  # 64 通道 -> 64x64
        #     nn.Conv2d(params['feature_chns'][1], class_num, 1),  # 32 通道 -> 128x128
        #     nn.Conv2d(params['feature_chns'][0], class_num, 1)   # 16 通道 -> 256x256
        # ])

    def forward(self, x, need_fp=False, contrastive_feature=False):
        feature = self.encoder(x)
        # feature[0]:torch.Size([12, 16, 256, 256])
        # feature[1]:torch.Size([12, 32, 128, 128])
        # feature[2]:torch.Size([12, 64, 64, 64])
        # feature[3]:torch.Size([12, 128, 32, 32])
        # feature[4]:torch.Size([12, 256, 16, 16])
        # 特征扰动模式
        
        if contrastive_feature:
            if need_fp:
                feature_decoder =  self.decoder([nn.Dropout2d(0.5)(feat) for feat in feature], is_contrastive=True)
            else:
                feature_decoder = self.decoder(feature, is_contrastive=True)
            return feature, feature_decoder
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)


        output = self.decoder(feature)
        return output
    
class UNet_multi_scale(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_multi_scale, self).__init__()

        # 参数配置（保持不变）
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        # 添加多尺度输出层
        self.scale_heads = nn.ModuleList([
            nn.Conv2d(params['feature_chns'][3], class_num, 1),  # 128 通道 -> 32x32
            nn.Conv2d(params['feature_chns'][2], class_num, 1),  # 64 通道 -> 64x64
            nn.Conv2d(params['feature_chns'][1], class_num, 1),  # 32 通道 -> 128x128
            nn.Conv2d(params['feature_chns'][0], class_num, 1)   # 16 通道 -> 256x256
        ])

    def forward(self, x, need_fp=False, contrastive_feature=False, multi_scale=False):
        # 编码器提取特征
        feature = self.encoder(x)
        # feature[0]: torch.Size([12, 16, 256, 256])
        # feature[1]: torch.Size([12, 32, 128, 128])
        # feature[2]: torch.Size([12, 64, 64, 64])
        # feature[3]: torch.Size([12, 128, 32, 32])
        # feature[4]: torch.Size([12, 256, 16, 16])

        # 返回对比特征（保持不变）
        if contrastive_feature:
            return feature

        # 特征扰动模式（保持不变）
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)

        # 多尺度输出模式
        if multi_scale:
            outputs = []
            # 从编码器特征图生成多尺度预测
            for i, head in enumerate(self.scale_heads):
                scale_pred = head(feature[3 - i])  # 从 feature[3] 到 feature[0]
                outputs.append(scale_pred)
            # 添加解码器的主输出
            main_output = self.decoder(feature)
            outputs.append(main_output)
            return outputs
        else:
            # 单尺度输出（原始行为）
            output = self.decoder(feature)
            return output



class UNet_unimatch_raw(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_unimatch, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        # 仅在高层特征上使用 SIREN
        # self.siren = SIRENLayer(256, 256)
        # self.siren = SIREN(in_features=256, hidden_features=256, hidden_layers=3, out_features=256)

        # self.se_blocks = nn.ModuleList([SEBlock(channels) for channels in [16, 32, 64, 128, 256]])  # 事先定义
        # self.fat_blocks = nn.ModuleList([FAT(channels) for channels in [16, 32, 64, 128, 256]])  # 事先定义
    def forward(self, x, need_fp=False):
        feature = self.encoder(x)
        # feature[0]:torch.Size([12, 16, 256, 256])
        # feature[1]:torch.Size([12, 32, 128, 128])
        # feature[2]:torch.Size([12, 64, 64, 64])
        # feature[3]:torch.Size([12, 128, 32, 32])
        # feature[4]:torch.Size([12, 256, 16, 16])
        # 特征扰动模式
        if need_fp:
            # outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            # outs = self.decoder([torch.cat((feat, se(feat))) for feat, se in zip(feature, self.se_blocks)])
            # feature[-1] = self.siren(feature[-1].flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(feature[-1])
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            # outs = self.decoder([torch.cat((feat, fat(feat))) for feat, fat in zip(feature, self.fat_blocks)])
            
            # torch.cat((feat, nn.Dropout2d(0.5)(feat))):
            # torch.cat()是沿batch维度拼接的。(这里AI认为是通道数拼接，实际上不是)
            # 假设feat的形状是[batch_size, channels, height, width]，
            # 那么经过nn.Dropout2d处理后，它会得到一个与原始feat形状相同的张量。
            # 拼接后的结果将变为[2*batch_size, channels, height, width]，即将feat和dropout后的feat沿batch维度拼接。
            # 如果outs的形状是[batch_size, channels, height, width]，
            # 那么它将被拆分成两个形状为[batch_size//2, channels, height, width]的块。
            return outs.chunk(2)


        output = self.decoder(feature)
        return output


class UNet_unimatch_reconstruction(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_unimatch_reconstruction, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.reconstruction_decoder = Decoder({**params, 'class_num': in_chns})  # 重建解码器
    def channel_exchange(self, feature, exchange_prob=0.2):
        B, C, H, W = feature.shape
        # assert B == 12, "Batch size must be 12"
        
        # 分割为前6和后6
        f1 = feature[:B//2].clone()  # [6, C, H, W]
        f2 = feature[B//2:].clone()  # [6, C, H, W]
        
        # 生成通道掩码
        mask = (torch.rand(C, device=feature.device) < exchange_prob)
        mask = mask.view(1, C, 1, 1).expand(B//2, C, H, W)  # 扩展到匹配维度
        
        # 交换选中通道的数据
        new_f1 = f1 * ~mask + f2 * mask
        new_f2 = f2 * ~mask + f1 * mask
        
        # 合并并返回
        return torch.cat([new_f1, new_f2], dim=0)
    
    def forward(self, x, need_fp=False, need_reconstruction=False, channel_exchange=False):
        feature = self.encoder(x)
        if channel_exchange:
            feature = [self.channel_exchange(feat) for feat in feature]

        if need_reconstruction:
            if need_fp:
                rec_output = self.reconstruction_decoder([nn.Dropout2d(0.5)(feat) for feat in feature])
                return rec_output
            rec_output = self.reconstruction_decoder(feature) 
            return rec_output

        # 特征扰动模式
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)


        output = self.decoder(feature)
        return output

## 以下是Unimatch版本
## https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/model/unet.py
class UNet_unimatch(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_unimatch, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
    def forward(self, x, need_fp=False):
        feature = self.encoder(x)
        # feature[0]:torch.Size([12, 16, 256, 256])
        # feature[1]:torch.Size([12, 32, 128, 128])
        # feature[2]:torch.Size([12, 64, 64, 64])
        # feature[3]:torch.Size([12, 128, 32, 32])
        # feature[4]:torch.Size([12, 256, 16, 16])
        # 特征扰动模式
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)

        output = self.decoder(feature)
        return output

class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "bilinear": False,
            "acti_func": "relu",
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.ema_decoder = None

        sparse_init_weight(self.encoder)
        sparse_init_weight(self.decoder)
        if self.ema_decoder is not None:
            sparse_init_weight(self.ema_decoder)

        self.pert = PertDropout(0.5).cuda()

    def forward(self, x, need_fp=False, need_ema=False, both=False, drop_rate=0.5):
        feature = self.encoder(x)

        if need_fp:
            features_x = []
            features_u = []
            for feats in feature:
                fx, fu = feats.chunk(2)
                features_x.append(fx)
                features_u.append(fu)

            perted_fus = self.pert(features_u)
            all_zip = zip(features_x, features_u, *perted_fus)
            outs = self.decoder([torch.cat(feats_all) for feats_all in all_zip])
            return outs.chunk(2 + len(self.pert))

        if need_ema:
            pert = (
                nn.FeatureAlphaDropout(0.5)
                if random.random() < 0.5
                else nn.AlphaDropout(0.5)
            )
            return self.decoder(feature), self.ema_decoder(
                [pert(feat) for feat in feature]
            )

        output = self.decoder(feature)
        return output



if __name__ == "__main__":
    model = UNet(1, 4).cuda()
    x = torch.randn((12, 1, 64, 64)).cuda()
    y = model(x, True)
    for i in y:
        print(i.shape)