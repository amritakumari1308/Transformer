------Hypergraph Transformer Multi-head Self-attention mechanism model----------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 🔥 HYPERGRAPH CONSTRUCTION
# ============================================================
class HypergraphConstruction:
    def __init__(self, k=10):
        self.k = k

    def construct_instance_hypergraph(self, X_list):
        n = X_list[0].shape[0]
        view = len(X_list)

        H_s = torch.zeros((n, view * n), device=X_list[0].device)

        for i in range(n):
            for v in range(view):
                H_s[i, v * n + i] = 1

        return H_s

    def construct_modality_hypergraph(self, X):
        n = X.shape[0]

        X = F.normalize(X, dim=1)
        sim = torch.mm(X, X.t())

        sim.fill_diagonal_(-float('inf'))
        _, idx = sim.topk(self.k, dim=1)

        H = torch.zeros((n, n), device=X.device)
        row = torch.arange(n).unsqueeze(1).expand(-1, self.k).to(X.device)
        H[row, idx] = 1

        return H

    def construct_full_hypergraph(self, X_list):
        H_s = self.construct_instance_hypergraph(X_list)
        H_m_list = [self.construct_modality_hypergraph(X) for X in X_list]
        H_m = torch.block_diag(*H_m_list)
        return H_s, H_m


# ============================================================
# 🔥 HGNN PROPAGATION (CORRECT)
# ============================================================
class HypergraphPropagation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.theta = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, X, H):
        Dv = torch.sum(H, dim=1)
        De = torch.sum(H, dim=0)

        Dv_inv_sqrt = torch.diag(1.0 / torch.sqrt(Dv + 1e-8))
        De_inv = torch.diag(1.0 / (De + 1e-8))

        return Dv_inv_sqrt @ H @ De_inv @ H.t() @ Dv_inv_sqrt @ X @ self.theta


# ============================================================
# 🔥 SAFE ATTENTION (NO MASK)
# ============================================================
class HypergraphAttention(nn.Module):
    def __init__(self, dim, dk):
        super().__init__()
        self.W_q = nn.Parameter(torch.Tensor(dim, dk))
        self.W_k = nn.Parameter(torch.Tensor(dim, dk))
        self.W_v = nn.Parameter(torch.Tensor(dim, dim))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, X, Y, H):
        Q = Y @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        scores = (Q @ K.t()) / (K.shape[1] ** 0.5)
        scores = scores.masked_fill(H.T == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        return attn @ V


# ============================================================
# 🔥 HYPERGRAPH BLOCK (FIXED)
# ============================================================
class HypergraphBlock(nn.Module):
    def __init__(self, dim, dk):
        super().__init__()
        self.attn = HypergraphAttention(dim, dk)
        self.prop = HypergraphPropagation(dim)

    def forward(self, X, Y_s, H_s, H_m):

        # Instance attention (uses H_s)
        Y_s_new = self.attn(X, Y_s, H_s.T)

        # Modality propagation (ONLY H_m)
        Y_m = self.prop(X, H_m)

        # Fusion attention
        X_new = self.attn(
            torch.cat([Y_s_new, Y_m], dim=0),
            X,
            torch.cat([H_s, H_m], dim=0)
        )

        return X_new, Y_s_new
# ============================================================
# 🔥 MULTI-LAYER HYPERGRAPH MODULE
# ============================================================
class HypergraphModule(nn.Module):
    def __init__(self, dim, dk=64, layers=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            HypergraphBlock(dim, dk) for _ in range(layers)
        ])

    def forward(self, X, X_list, H_s, H_m):

        Y_s = sum(X_list) / len(X_list)

        X_out, Y_s = self.blocks[0](X, Y_s, H_s, H_m)

        for i in range(1, len(self.blocks)):
            X_out, Y_s = self.blocks[i](X_out, Y_s, H_s, H_m)

        zh = torch.split(X_out, X_list[0].shape[0], dim=0)

        return zh


# ============================================================
# 🔥 TRANSFORMER FUSION (LEARNABLE)
# ============================================================
class TransformerFusion(nn.Module):
    def __init__(self, dim, num_modalities, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

        self.alpha = nn.Parameter(torch.ones(num_modalities))

    def forward(self, z_list):
        Z = torch.stack(z_list, dim=1)

        attn_out, _ = self.attn(Z, Z, Z)
        Z = self.norm1(Z + attn_out)

        ff = self.ffn(Z)
        Z = self.norm2(Z + ff)

        weights = torch.softmax(self.alpha, dim=0)
        Z = Z * weights.unsqueeze(0).unsqueeze(-1)

        return Z.sum(dim=1)


# ============================================================
# 🔥 FINAL NETWORK
# ============================================================
class Network(nn.Module):
    def __init__(self, input_dims, feature_dim, num_classes):
        super().__init__()

        self.view = len(input_dims)

        # Encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[v], 512),
                nn.ReLU(),
                nn.Linear(512, feature_dim)
            ) for v in range(self.view)
        ])

        # Decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, input_dims[v])
            ) for v in range(self.view)
        ])

        self.hyper_builder = HypergraphConstruction(k=10)

        self.hypergraph = HypergraphModule(
            dim=feature_dim,
            dk=64,
            layers=3
        )

        self.fusion = TransformerFusion(
            dim=feature_dim * 2,
            num_modalities=self.view
        )

        self.classifier = nn.Linear(feature_dim * 2, num_classes)

    def forward(self, xs):

        zs = []
        xrs = []

        # Encode
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            zs.append(z)

        # Hypergraph
        X = torch.cat(zs, dim=0)
        H_s, H_m = self.hyper_builder.construct_full_hypergraph(zs)

        zh = self.hypergraph(X, zs, H_s, H_m)

        # Decode using zh (correct)
        for v in range(self.view):
            xrs.append(self.decoders[v](zh[v]))

        # Fusion
        fused = []
        for v in range(self.view):
            fused.append(torch.cat([zh[v], zs[v]], dim=1))

        Z = self.fusion(fused)

        logits = self.classifier(Z)

        return logits, xrs, Z
