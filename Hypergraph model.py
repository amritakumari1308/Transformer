
# ============================================================
# S-1
# DEVICE
# ============================================================

import torch
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# ============================================================
# S-2
# LOAD DATASETS
# ============================================================

import pandas as pd
import os

# IMPORTANT:
# Change only this path if your actual folder name is different.
BASE_DIR = "/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/survival/SURVIVAL_1035"

print("\nChecking data directory:")
print(BASE_DIR)

if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(
        f"\nDataset directory not found:\n{BASE_DIR}\n\n"
        "Check the folder name in Google Drive."
    )

print("Directory exists.")

cnv_data = pd.read_csv(
    f"{BASE_DIR}/CNV 1035.csv",
    header=None
).dropna(how="all")

clinical_data = pd.read_csv(
    f"{BASE_DIR}/CLINICAL 1035.csv",
    header=None
)

mRNA_data = pd.read_csv(
    f"{BASE_DIR}/mRNA 1035.csv",
    header=None
)

dna_data = pd.read_csv(
    f"{BASE_DIR}/DNA 1035.csv",
    header=None
)

mirna_data = pd.read_csv(
    f"{BASE_DIR}/miRNA 1035.csv",
    header=None
)

wsi_data = pd.read_csv(
    f"{BASE_DIR}/WSI 1035.csv",
    header=None
)

print("\nAll datasets loaded successfully.")


# ============================================================
# S-3
# SHAPES
# ============================================================

print("\n===== DATASET SHAPES =====")

print("Clinical:", clinical_data.shape)
print("CNV:", cnv_data.shape)
print("mRNA:", mRNA_data.shape)
print("DNA:", dna_data.shape)
print("miRNA:", mirna_data.shape)
print("WSI:", wsi_data.shape)


# ============================================================
# S-4
# LABELS
# ============================================================

clinical_ids_all = (
    clinical_data.iloc[:, 0]
    .astype(str)
    .values
)

clinical_tmp = clinical_data.iloc[:, 1:].copy()

clinical_tmp = clinical_tmp.apply(
    pd.to_numeric,
    errors="coerce"
)

clinical_tmp = clinical_tmp.dropna(
    axis=1,
    how="all"
)

# Last available clinical column is used as label
y = clinical_tmp.iloc[:, -1].values

valid_label_mask = ~pd.isna(y)

y = y[valid_label_mask].astype(int)

clinical_ids = clinical_ids_all[
    valid_label_mask
]

print("\n===== LABEL DISTRIBUTION =====")
print(pd.Series(y).value_counts())


# ============================================================
# S-5
# DATA PREVIEW
# ============================================================

print("\n===== CNV PREVIEW =====")
print(cnv_data.head())


# ============================================================

# S-7
# LEAKAGE-SAFE PREPROCESSOR
# ============================================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class LeakageSafePreprocessor:

    def __init__(
        self,
        n_components,
        name,
        use_scaler=True,
        scaler_type="standard"
    ):

        self.n_components = n_components
        self.name = name
        self.use_scaler = use_scaler
        self.scaler_type = scaler_type

        self.imputer_columns = None
        self.keep_mask = None
        self.imputer = None
        self.scaler = None
        self.pca = None

    def fit(self, X_train):

        X_train = pd.DataFrame(X_train).copy()

        X_train = X_train.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # --------------------------------------------------------
        # Column selection learned ONLY from TRAIN
        # --------------------------------------------------------

        self.imputer_columns = [
            c
            for c in X_train.columns
            if not X_train[c].isna().all()
        ]

        X_train = X_train[
            self.imputer_columns
        ]

        # --------------------------------------------------------
        # Median imputation fitted ONLY on TRAIN
        # --------------------------------------------------------

        self.imputer = SimpleImputer(
            strategy="median"
        )

        X_train = self.imputer.fit_transform(
            X_train
        )

        # --------------------------------------------------------
        # Constant-column removal learned ONLY from TRAIN
        # --------------------------------------------------------

        std_train = np.std(
            X_train,
            axis=0
        )

        self.keep_mask = (
            std_train > 0
        )

        X_train = X_train[
            :,
            self.keep_mask
        ]

        # --------------------------------------------------------
        # Scaling fitted ONLY on TRAIN
        # --------------------------------------------------------

        if self.use_scaler:

            if self.scaler_type == "standard":

                self.scaler = StandardScaler()

            elif self.scaler_type == "minmax":

                self.scaler = MinMaxScaler()

            else:

                raise ValueError(
                    "Unknown scaler_type"
                )

            X_train = self.scaler.fit_transform(
                X_train
            )

        # --------------------------------------------------------
        # PCA fitted ONLY on TRAIN
        # --------------------------------------------------------

        max_components = min(
            self.n_components,
            X_train.shape[0],
            X_train.shape[1]
        )

        self.pca = PCA(
            n_components=max_components,
            svd_solver="auto",
            random_state=42
        )

        X_train = self.pca.fit_transform(
            X_train
        )

        X_train = np.nan_to_num(
            X_train
        ).astype(
            np.float32
        )

        return X_train

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # --------------------------------------------------------
        # Use TRAIN-LEARNED columns
        # --------------------------------------------------------

        X = X.reindex(
            columns=self.imputer_columns
        )

        # --------------------------------------------------------
        # TRAIN-FITTED imputer
        # --------------------------------------------------------

        X = self.imputer.transform(
            X
        )

        # --------------------------------------------------------
        # TRAIN-LEARNED constant-column mask
        # --------------------------------------------------------

        X = X[
            :,
            self.keep_mask
        ]

        # --------------------------------------------------------
        # TRAIN-FITTED scaler
        # --------------------------------------------------------

        if self.use_scaler:

            X = self.scaler.transform(
                X
            )

        # --------------------------------------------------------
        # TRAIN-FITTED PCA
        # --------------------------------------------------------

        X = self.pca.transform(
            X
        )

        X = np.nan_to_num(
            X
        ).astype(
            np.float32
        )

        return X


# ============================================================

# S-12
# INDUCTIVE HYPERGRAPH
# ============================================================

import torch.nn as nn
import torch.nn.functional as F


class HypergraphConstruction:

    def __init__(
        self,
        k=10
    ):

        self.k = k


    # --------------------------------------------------------
    # TRAINING PATIENTS ONLY
    # Hs:
    # modality-node x patient-node
    # --------------------------------------------------------

    def construct_instance_hypergraph(
        self,
        X_list
    ):

        n = X_list[0].shape[0]

        v = len(X_list)

        H_s = torch.zeros(
            (
                v * n,
                n
            ),
            device=X_list[0].device
        )


        for modality in range(v):

            start = modality * n

            end = (
                modality + 1
            ) * n


            rows = torch.arange(
                start,
                end,
                device=X_list[0].device
            )


            cols = torch.arange(
                n,
                device=X_list[0].device
            )


            H_s[
                rows,
                cols
            ] = 1.0


        return H_s


    # --------------------------------------------------------
    # MODALITY-SPECIFIC HYPERGRAPH
    # TRAINING PATIENTS ONLY
    # --------------------------------------------------------

    def construct_modality_hypergraph(
        self,
        X
    ):

        n = X.shape[0]


        if n <= 1:

            return torch.eye(
                n,
                device=X.device
            )


        k = min(
            self.k,
            n - 1
        )


        X_norm = F.normalize(
            X,
            dim=1
        )


        sim = torch.mm(
            X_norm,
            X_norm.t()
        )


        sim.fill_diagonal_(
            -float("inf")
        )


        _, idx = sim.topk(
            k,
            dim=1
        )


        H = torch.zeros(
            (
                n,
                n
            ),
            device=X.device
        )


        row = torch.arange(
            n,
            device=X.device
        ).unsqueeze(
            1
        ).expand(
            -1,
            k
        )


        H[
            row,
            idx
        ] = 1.0


        return H


    # --------------------------------------------------------
    # COMPLETE TRAINING HYPERGRAPH
    # --------------------------------------------------------

    def construct_full_hypergraph(
        self,
        X_list,
        patient_mask=None
    ):

        # STRICT INDUCTIVE MASK: filter patients BEFORE any
        # cosine-similarity, kNN, Hs, Hm, or message-passing operation.
        if len(X_list) == 0:
            raise ValueError("X_list cannot be empty.")

        n_total = X_list[0].shape[0]

        for X in X_list:
            if X.shape[0] != n_total:
                raise ValueError(
                    "All modalities must contain the same number of patients."
                )

        if patient_mask is None:
            patient_mask = torch.ones(
                n_total, dtype=torch.bool, device=X_list[0].device
            )
        else:
            patient_mask = torch.as_tensor(
                patient_mask, dtype=torch.bool, device=X_list[0].device
            )
            if patient_mask.ndim != 1 or patient_mask.shape[0] != n_total:
                raise ValueError(
                    "patient_mask must have shape [number_of_patients]."
                )

        if patient_mask.sum().item() < 1:
            raise ValueError("patient_mask must contain at least one patient.")

        # Excluded patients are physically removed before graph construction.
        X_graph = [X[patient_mask] for X in X_list]

        n_graph = X_graph[0].shape[0]
        if n_graph != int(patient_mask.sum().item()):
            raise RuntimeError(
                "Patient mask was not applied before hypergraph construction."
            )

        H_s = self.construct_instance_hypergraph(X_graph)

        H_m_list = [
            self.construct_modality_hypergraph(X)
            for X in X_graph
        ]

        H_m = torch.block_diag(*H_m_list)

        return H_s, H_m, X_graph, patient_mask


# ============================================================
# HYPERGRAPH PROPAGATION
# ============================================================

class HypergraphPropagation(
    nn.Module
):

    def __init__(
        self,
        dim
    ):

        super().__init__()


        self.theta = nn.Parameter(
            torch.Tensor(
                dim,
                dim
            )
        )


        nn.init.xavier_uniform_(
            self.theta
        )


    def forward(
        self,
        X,
        H
    ):

        Dv = torch.sum(
            H,
            dim=1
        )


        De = torch.sum(
            H,
            dim=0
        )


        Dv_inv_sqrt = torch.diag(
            1.0 /
            torch.sqrt(
                Dv + 1e-8
            )
        )


        De_inv = torch.diag(
            1.0 /
            (
                De + 1e-8
            )
        )


        return (
            Dv_inv_sqrt
            @ H
            @ De_inv
            @ H.t()
            @ Dv_inv_sqrt
            @ X
            @ self.theta
        )


# ============================================================
# HYPERGRAPH ATTENTION
# ============================================================

class HypergraphAttention(
    nn.Module
):

    def __init__(
        self,
        dim,
        dk
    ):

        super().__init__()


        self.W_q = nn.Parameter(
            torch.Tensor(
                dim,
                dk
            )
        )


        self.W_k = nn.Parameter(
            torch.Tensor(
                dim,
                dk
            )
        )


        self.W_v = nn.Parameter(
            torch.Tensor(
                dim,
                dim
            )
        )


        nn.init.xavier_uniform_(
            self.W_q
        )

        nn.init.xavier_uniform_(
            self.W_k
        )

        nn.init.xavier_uniform_(
            self.W_v
        )


    def forward(
        self,
        X,
        Y,
        H
    ):

        Q = Y @ self.W_q

        K = X @ self.W_k

        V = X @ self.W_v


        scores = (
            Q @ K.t()
        ) / (
            K.shape[1] ** 0.5
        )


        if H is not None:

            if H.shape != scores.shape:

                raise ValueError(
                    f"Attention mask shape "
                    f"{H.shape} does not match "
                    f"score shape "
                    f"{scores.shape}"
                )


            scores = scores.masked_fill(
                H == 0,
                float("-inf")
            )


        attn = torch.softmax(
            scores,
            dim=-1
        )


        return attn @ V


# ============================================================
# HYPERGRAPH BLOCK
# ============================================================

class HypergraphBlock(
    nn.Module
):

    def __init__(
        self,
        dim,
        dk
    ):

        super().__init__()


        self.attn = HypergraphAttention(
            dim,
            dk
        )


        self.prop = HypergraphPropagation(
            dim
        )


    def forward(
        self,
        X,
        Y_s,
        H_s,
        H_m
    ):

        Y_s_new = self.attn(
            X,
            Y_s,
            H_s.t()
        )


        Y_m = self.prop(
            X,
            H_m
        )


        context = torch.cat(
            [
                Y_s_new,
                Y_m
            ],
            dim=0
        )


        context_mask = torch.cat(
            [
                H_s,
                H_m
            ],
            dim=1
        )


        X_new = self.attn(
            context,
            X,
            context_mask
        )


        return (
            X_new,
            Y_s_new
        )


# ============================================================
# MULTI-LAYER HYPERGRAPH MODULE
# ============================================================

class HypergraphModule(
    nn.Module
):

    def __init__(
        self,
        dim,
        dk=64,
        layers=3
    ):

        super().__init__()


        self.blocks = nn.ModuleList(
            [
                HypergraphBlock(
                    dim,
                    dk
                )

                for _ in range(layers)
            ]
        )


    def forward(
        self,
        X,
        X_list,
        H_s,
        H_m
    ):

        Y_s = (
            sum(X_list)
            /
            len(X_list)
        )


        X_out, Y_s = (
            self.blocks[0](
                X,
                Y_s,
                H_s,
                H_m
            )
        )


        for i in range(
            1,
            len(self.blocks)
        ):

            X_out, Y_s = (
                self.blocks[i](
                    X_out,
                    Y_s,
                    H_s,
                    H_m
                )
            )


        n = X_list[0].shape[0]


        zh = torch.split(
            X_out,
            n,
            dim=0
        )


        return zh


# ============================================================
# TRANSFORMER MULTI-HEAD SELF-ATTENTION FUSION
# ============================================================

class TransformerFusion(
    nn.Module
):

    def __init__(
        self,
        dim,
        num_modalities,
        heads=4
    ):

        super().__init__()


        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )


        self.norm1 = nn.LayerNorm(
            dim
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                dim,
                dim * 2
            ),

            nn.ReLU(),

            nn.Linear(
                dim * 2,
                dim
            )

        )


        self.norm2 = nn.LayerNorm(
            dim
        )


        self.alpha = nn.Parameter(
            torch.ones(
                num_modalities
            )
        )


    def forward(
        self,
        z_list
    ):

        Z = torch.stack(
            z_list,
            dim=1
        )


        attn_out, _ = self.attn(
            Z,
            Z,
            Z
        )


        Z = self.norm1(
            Z + attn_out
        )


        ff = self.ffn(
            Z
        )


        Z = self.norm2(
            Z + ff
        )


        weights = torch.softmax(
            self.alpha,
            dim=0
        )


        Z = (
            Z
            *
            weights.unsqueeze(0)
            .unsqueeze(-1)
        )


        return Z.sum(
            dim=1
        )


# ============================================================
# FINAL NETWORK
# STRICTLY INDUCTIVE HYPERGRAPH  DESIGN
# ============================================================

class InductiveHypergraphAdapter(nn.Module):
    """
    Learnable patient-wise adapter used for inductive inference.

    It never constructs Hs/Hm and never performs hypergraph
    message passing. Thus unseen validation/test patients do not
    enter the training hypergraph.
    """

    def __init__(self, dim):

        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(
                dim,
                dim * 2
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(
                dim * 2,
                dim
            )
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, z):

        return self.norm(
            z + self.adapter(z)
        )


class Network(
    nn.Module
):

    def __init__(
        self,
        input_dims,
        feature_dim,
        num_classes
    ):

        super().__init__()

        self.view = len(
            input_dims
        )

        # --------------------------------------------------------
        # MODALITY ENCODERS
        # --------------------------------------------------------

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        input_dims[v],
                        512
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        512,
                        feature_dim
                    )
                )
                for v in range(
                    self.view
                )
            ]
        )

        # --------------------------------------------------------
        # MODALITY DECODERS
        # --------------------------------------------------------

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        feature_dim,
                        512
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        512,
                        input_dims[v]
                    )
                )
                for v in range(
                    self.view
                )
            ]
        )

        # --------------------------------------------------------
        # TRAINING-ONLY DUAL HYPERGRAPH
        # --------------------------------------------------------

        self.hyper_builder = (
            HypergraphConstruction(
                k=10
            )
        )

        self.hypergraph = (
            HypergraphModule(
                dim=feature_dim,
                dk=64,
                layers=3
            )
        )

        # --------------------------------------------------------
        # INDUCTIVE HYPERGRAPH ADAPTER
        #
        # This path is used for training, validation and test.
        # It never constructs a hypergraph.
        # --------------------------------------------------------

        self.inductive_adapter = nn.ModuleList(
            [
                InductiveHypergraphAdapter(
                    feature_dim
                )
                for _ in range(
                    self.view
                )
            ]
        )

        # --------------------------------------------------------
        # TRANSFORMER MULTI-HEAD FUSION
        # --------------------------------------------------------

        self.fusion = (
            TransformerFusion(
                dim=feature_dim * 2,
                num_modalities=self.view,
                heads=4
            )
        )

        self.classifier = nn.Linear(
            feature_dim * 2,
            num_classes
        )

    def encode(
        self,
        xs
    ):

        zs = []

        for v in range(
            self.view
        ):

            z = self.encoders[v](
                xs[v]
            )

            zs.append(
                z
            )

        return zs

    def forward(
        self,
        xs,
        graph_mode=False,
        return_hypergraph=False
    ):

        # --------------------------------------------------------
        # MODALITY ENCODING
        # --------------------------------------------------------

        zs = self.encode(
            xs
        )

        # --------------------------------------------------------
        # INDUCTIVE REPRESENTATION
        #
        # Always available. This is the representation used by
        # the classifier for train/validation/test.
        # --------------------------------------------------------

        zh_inductive = [
            self.inductive_adapter[v](
                zs[v]
            )
            for v in range(
                self.view
            )
        ]

        zh_hypergraph = None

        # --------------------------------------------------------
        # 
        #
        # graph_mode=True is called ONLY with xs_train.
        # Validation/test calls use graph_mode=False.
        # --------------------------------------------------------

        if graph_mode:

            train_patient_mask = torch.ones(
                zs[0].shape[0],
                dtype=torch.bool,
                device=zs[0].device
            )

            H_s, H_m, zs_graph, _ = (
                self.hyper_builder
                .construct_full_hypergraph(
                    zs,
                    patient_mask=train_patient_mask
                )
            )

            X_graph = torch.cat(
                zs_graph,
                dim=0
            )

            zh_hypergraph = self.hypergraph(
                X_graph,
                zs_graph,
                H_s,
                H_m
            )

        # --------------------------------------------------------
        # RECONSTRUCTION
        #
        # Reconstruction is based on the inductive representation
        # so that the representation trained here is also valid for
        # unseen patients.
        # --------------------------------------------------------

        xrs = []

        for v in range(
            self.view
        ):

            xrs.append(
                self.decoders[v](
                    zh_inductive[v]
                )
            )

        # --------------------------------------------------------
        # INDUCTIVE FUSION
        # --------------------------------------------------------

        fused_inductive = []

        for v in range(
            self.view
        ):

            fused_inductive.append(
                torch.cat(
                    [
                        zh_inductive[v],
                        zs[v]
                    ],
                    dim=1
                )
            )

        Z_inductive = self.fusion(
            fused_inductive
        )

        logits_inductive = self.classifier(
            Z_inductive
        )

        # --------------------------------------------------------
       
        #
        # Created only during training. This makes the actual
        # hypergraph representation discriminative while the
        # inductive adapter learns to reproduce it.
        # --------------------------------------------------------

        logits_hypergraph = None
        Z_hypergraph= None

        if zh_hypergraph is not None:

            fused_hypergraph = []

            for v in range(
                self.view
            ):

                fused_hypergraph.append(
                    torch.cat(
                        [
                            zh_hypergraph[v],
                            zs[v]
                        ],
                        dim=1
                    )
                )

            Z_hypergraph = self.fusion(
                fused_hypergraph
            )

            logits_hypergraph = self.classifier(
                Z_hypergraph
            )

        if return_hypergraph:

            return (
                logits_inductive,
                logits_hypergraph,
                xrs,
                Z_inductive,
                Z_hypergraph,
                zh_inductive,
                zh_hypergraph
            )

        return (
            logits_inductive,
            xrs,
            Z_inductive
        )


# ============================================================
# ============================================================
# S-6
# 10-FOLD OUTER STRATIFIED CROSS-VALIDATION
#
# OUTER TEST FOLD IS NEVER USED FOR:
#   - IMPUTATION FITTING
#   - SCALING FITTING
#   - PCA FITTING
#   - HYPERGRAPH CONSTRUCTION
#   - MODEL TRAINING
#   - THRESHOLD SELECTION
#
# INNER VALIDATION IS USED ONLY FOR:
#   - EARLY STOPPING
#   - LR SCHEDULING
#   - THRESHOLD SELECTION
# ============================================================

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve
)
import copy
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Align every modality to the SAME patient order
# ------------------------------------------------------------

def filter_by_ids(df, ids):
    df = df.copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

    mapping = {
        str(pid): i for i, pid in enumerate(df.iloc[:, 0].values)
    }

    missing = [pid for pid in ids if str(pid) not in mapping]

    if len(missing) > 0:
        raise ValueError(
            f"Patients missing from modality: {missing[:10]}"
        )

    order = [mapping[str(pid)] for pid in ids]

    return df.iloc[order].reset_index(drop=True)


cnv_data = filter_by_ids(cnv_data, clinical_ids)
mRNA_data = filter_by_ids(mRNA_data, clinical_ids)
dna_data = filter_by_ids(dna_data, clinical_ids)
mirna_data = filter_by_ids(mirna_data, clinical_ids)
wsi_data = filter_by_ids(wsi_data, clinical_ids)

assert np.array_equal(
    clinical_ids,
    cnv_data.iloc[:, 0].astype(str).values
)
assert np.array_equal(
    clinical_ids,
    mRNA_data.iloc[:, 0].astype(str).values
)
assert np.array_equal(
    clinical_ids,
    dna_data.iloc[:, 0].astype(str).values
)
assert np.array_equal(
    clinical_ids,
    mirna_data.iloc[:, 0].astype(str).values
)
assert np.array_equal(
    clinical_ids,
    wsi_data.iloc[:, 0].astype(str).values
)

print("\nPatient alignment verified.")
print("Total patients:", len(y))
print("Label distribution:")
print(pd.Series(y).value_counts())

# ============================================================
# S-7
# LEAKAGE-SAFE PREPROCESSOR
# ============================================================

class LeakageSafePreprocessor:

    def __init__(
        self,
        n_components,
        name,
        use_scaler=True,
        scaler_type="standard"
    ):
        self.n_components = n_components
        self.name = name
        self.use_scaler = use_scaler
        self.scaler_type = scaler_type

        self.imputer_columns = None
        self.keep_mask = None
        self.imputer = None
        self.scaler = None
        self.pca = None

    def fit(self, X_train):

        X_train = pd.DataFrame(X_train).copy()

        X_train = X_train.apply(
            pd.to_numeric,
            errors="coerce"
        )

        self.imputer_columns = [
            c for c in X_train.columns
            if not X_train[c].isna().all()
        ]

        X_train = X_train[self.imputer_columns]

        self.imputer = SimpleImputer(strategy="median")
        X_train = self.imputer.fit_transform(X_train)

        std_train = np.std(X_train, axis=0)
        self.keep_mask = std_train > 0

        X_train = X_train[:, self.keep_mask]

        if self.use_scaler:

            if self.scaler_type == "standard":
                self.scaler = StandardScaler()

            elif self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()

            else:
                raise ValueError("Unknown scaler_type")

            X_train = self.scaler.fit_transform(X_train)

        max_components = min(
            self.n_components,
            X_train.shape[0],
            X_train.shape[1]
        )

        self.pca = PCA(
            n_components=max_components,
            svd_solver="auto",
            random_state=42
        )

        X_train = self.pca.fit_transform(X_train)

        X_train = np.nan_to_num(X_train).astype(np.float32)

        return X_train

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.reindex(columns=self.imputer_columns)

        X = self.imputer.transform(X)

        X = X[:, self.keep_mask]

        if self.use_scaler:
            X = self.scaler.transform(X)

        X = self.pca.transform(X)

        X = np.nan_to_num(X).astype(np.float32)

        return X


def get_feature_matrix(df):
    return df.iloc[:, 1:].copy()


def preprocess_cnv(df):

    X = df.iloc[:, 1:].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # CNV preprocessing is deterministic and uses no cohort-derived statistic.
    X = X.fillna(0)
    X = X.clip(-2, 2)

    return X.values.astype(np.float32)


def build_fold_data(train_idx, val_idx, test_idx):

    # --------------------------------------------------------
    # OMICS: FIT ONLY ON OUTER-TRAIN / INNER-TRAIN
    # --------------------------------------------------------

    mRNA_preprocessor = LeakageSafePreprocessor(
        n_components=500,
        name="mRNA",
        use_scaler=True,
        scaler_type="standard"
    )

    dna_preprocessor = LeakageSafePreprocessor(
        n_components=500,
        name="DNA",
        use_scaler=True,
        scaler_type="standard"
    )

    mirna_preprocessor = LeakageSafePreprocessor(
        n_components=300,
        name="miRNA",
        use_scaler=True,
        scaler_type="standard"
    )

    wsi_preprocessor = LeakageSafePreprocessor(
        n_components=200,
        name="WSI",
        use_scaler=False
    )

    X_mRNA_train = mRNA_preprocessor.fit(
        get_feature_matrix(mRNA_data.iloc[train_idx])
    )
    X_mRNA_val = mRNA_preprocessor.transform(
        get_feature_matrix(mRNA_data.iloc[val_idx])
    )
    X_mRNA_test = mRNA_preprocessor.transform(
        get_feature_matrix(mRNA_data.iloc[test_idx])
    )

    X_dna_train = dna_preprocessor.fit(
        get_feature_matrix(dna_data.iloc[train_idx])
    )
    X_dna_val = dna_preprocessor.transform(
        get_feature_matrix(dna_data.iloc[val_idx])
    )
    X_dna_test = dna_preprocessor.transform(
        get_feature_matrix(dna_data.iloc[test_idx])
    )

    X_mirna_train = mirna_preprocessor.fit(
        get_feature_matrix(mirna_data.iloc[train_idx])
    )
    X_mirna_val = mirna_preprocessor.transform(
        get_feature_matrix(mirna_data.iloc[val_idx])
    )
    X_mirna_test = mirna_preprocessor.transform(
        get_feature_matrix(mirna_data.iloc[test_idx])
    )

    X_wsi_train = wsi_preprocessor.fit(
        get_feature_matrix(wsi_data.iloc[train_idx])
    )
    X_wsi_val = wsi_preprocessor.transform(
        get_feature_matrix(wsi_data.iloc[val_idx])
    )
    X_wsi_test = wsi_preprocessor.transform(
        get_feature_matrix(wsi_data.iloc[test_idx])
    )

    # --------------------------------------------------------
    # CLINICAL: FIT ONLY ON TRAIN
    # --------------------------------------------------------

    clinical_features = clinical_data.iloc[:, 1:].copy()
    clinical_features = clinical_features.apply(
        pd.to_numeric,
        errors="coerce"
    )
    clinical_features = clinical_features.dropna(
        axis=1,
        how="all"
    )

    clinical_X = clinical_features.iloc[:, :-1]

    cli_imputer = SimpleImputer(strategy="median")
    cli_scaler = MinMaxScaler()

    X_cli_train = cli_imputer.fit_transform(
        clinical_X.iloc[train_idx]
    )
    X_cli_train = cli_scaler.fit_transform(
        X_cli_train
    ).astype(np.float32)

    X_cli_val = cli_imputer.transform(
        clinical_X.iloc[val_idx]
    )
    X_cli_val = cli_scaler.transform(
        X_cli_val
    ).astype(np.float32)

    X_cli_test = cli_imputer.transform(
        clinical_X.iloc[test_idx]
    )
    X_cli_test = cli_scaler.transform(
        X_cli_test
    ).astype(np.float32)

    # --------------------------------------------------------
    # CNV: deterministic preprocessing
    # --------------------------------------------------------

    X_cnv_all = preprocess_cnv(cnv_data)

    X_cnv_train = X_cnv_all[train_idx]
    X_cnv_val = X_cnv_all[val_idx]
    X_cnv_test = X_cnv_all[test_idx]

    X_train_dict = {
        "mRNA": X_mRNA_train,
        "DNA": X_dna_train,
        "miRNA": X_mirna_train,
        "CNV": X_cnv_train,
        "WSI": X_wsi_train
    }

    X_val_dict = {
        "mRNA": X_mRNA_val,
        "DNA": X_dna_val,
        "miRNA": X_mirna_val,
        "CNV": X_cnv_val,
        "WSI": X_wsi_val
    }

    X_test_dict = {
        "mRNA": X_mRNA_test,
        "DNA": X_dna_test,
        "miRNA": X_mirna_test,
        "CNV": X_cnv_test,
        "WSI": X_wsi_test
    }

    # --------------------------------------------------------
    # --------------------------------------------------------
    # CONVERT FOLD-PROCESSED FEATURES TO TENSORS
    #
    # No second scaler is applied here.
    # Each modality preprocessor already fits its scaler ONLY
    # on the training split and transforms validation/test.
    # --------------------------------------------------------

    xs_train = []
    xs_val = []
    xs_test = []

    for name in X_train_dict.keys():

        Xt = np.asarray(
            X_train_dict[name],
            dtype=np.float32
        )

        Xv = np.asarray(
            X_val_dict[name],
            dtype=np.float32
        )

        Xte = np.asarray(
            X_test_dict[name],
            dtype=np.float32
        )

        xs_train.append(
            torch.tensor(
                Xt,
                dtype=torch.float32,
                device=device
            )
        )

        xs_val.append(
            torch.tensor(
                Xv,
                dtype=torch.float32,
                device=device
            )
        )

        xs_test.append(
            torch.tensor(
                Xte,
                dtype=torch.float32,
                device=device
            )
        )

    return (
        xs_train,
        xs_val,
        xs_test,
        X_cli_train,
        X_cli_val,
        X_cli_test
    )


# ============================================================
# S-8
# 10-FOLD TRAINING / VALIDATION / TEST
# ============================================================

N_SPLITS = 10
INNER_VAL_SIZE = 0.20
RANDOM_STATE = 42

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_metrics = []
all_test_predictions = []

print("\n")
print("=" * 80)
print("10-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 80)
print("Reviewer-safe settings:")
print("  - Imputation fitted on training split only")
print("  - Scaling fitted on training split only")
print("  - PCA fitted on training split only")
print("  - Hs/Hm constructed from training patients only")
print("  - Validation/test patients never enter hypergraph message passing")
print("  - Same inductive representation pathway used for train/val/test")
print("  - Binary classification evaluation retained")
print("=" * 80)

for fold, (outer_train_idx, outer_test_idx) in enumerate(
    skf.split(np.zeros(len(y)), y),
    start=1
):

    print("\n")
    print("=" * 80)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 80)

    # --------------------------------------------------------
    # INNER VALIDATION SPLIT
    # ONLY FROM OUTER TRAIN
    # --------------------------------------------------------

    inner_train_idx, val_idx = train_test_split(
        outer_train_idx,
        test_size=INNER_VAL_SIZE,
        stratify=y[outer_train_idx],
        random_state=RANDOM_STATE + fold
    )

    train_idx = np.asarray(inner_train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(outer_test_idx)

    print(
        f"Train: {len(train_idx)} | "
        f"Validation: {len(val_idx)} | "
        f"Test: {len(test_idx)}"
    )

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # --------------------------------------------------------
    # FOLD-SPECIFIC PREPROCESSING
    # --------------------------------------------------------

    (
        xs_train,
        xs_val,
        xs_test,
        X_cli_train,
        X_cli_val,
        X_cli_test
    ) = build_fold_data(
        train_idx,
        val_idx,
        test_idx
    )

    print("\nFold shapes:")

    for i, x in enumerate(xs_train):
        print(
            f"View {i}: "
            f"Train={tuple(xs_train[i].shape)} | "
            f"Val={tuple(xs_val[i].shape)} | "
            f"Test={tuple(xs_test[i].shape)}"
        )

    print(
        "Clinical:",
        X_cli_train.shape,
        X_cli_val.shape,
        X_cli_test.shape
    )

    # --------------------------------------------------------
    # LABEL TENSORS
    # --------------------------------------------------------

    y_train_tensor = torch.tensor(
        y[train_idx],
        dtype=torch.long,
        device=device
    )

    y_val_tensor = torch.tensor(
        y[val_idx],
        dtype=torch.long,
        device=device
    )

    y_test_tensor = torch.tensor(
        y[test_idx],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # TRAIN ONLY
    # --------------------------------------------------------

    classes_train = np.unique(y[train_idx])

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes_train,
        y=y[train_idx]
    )

    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Network(
        input_dims=[x.shape[1] for x in xs_train],
        feature_dim=128,
        num_classes=len(np.unique(y))
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    EPOCHS = 100
    PATIENCE = 15

    best_val_acc = -np.inf
    counter = 0
    best_model_state = None

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        # ----------------------------------------------------
        # TRAINING
        #
        # ONLY xs_train enters hypergraph construction.
        # Validation/test patients never enter Hs/Hm.
        # ----------------------------------------------------

        (
            logits_train,
            logits_hypergraph,
            xrs_train,
            _,
            _,
            zh_inductive_train,
            zh_hypergraph_train
        ) = model(
            xs_train,
            graph_mode=True,
            return_hypergraph=True
        )

        # Main loss on the SAME inductive representation used
        # during validation and test.
        loss_cls_inductive = criterion(
            logits_train,
            y_train_tensor
        )

        
       
        loss_cls_hypergraph = criterion(
            logits_hypergraph,
            y_train_tensor
        )

        # Reconstruction on the inductive representation.
        loss_rec = sum(
            F.mse_loss(
                xrs_train[v],
                xs_train[v]
            )
            for v in range(
                len(xs_train)
            )
        )

        # Align inductive representation with the actual
        # training-only hypergraph-enhanced representation.
        loss_hg = sum(
            F.mse_loss(
                zh_inductive_train[v],
                zh_hypergraph_train[v].detach()
            )
            for v in range(
                len(xs_train)
            )
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
            loss_cls_inductive
            + 0.30 * loss_cls_hypergraph
            + 0.05 * loss_hg
        )

        if epoch >= 10:
            loss = loss + 0.01 * loss_rec

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        with torch.no_grad():

            pred_train = logits_train.argmax(dim=1)

            train_acc = (
                pred_train == y_train_tensor
            ).float().mean().item()

            pred_hypergraph = logits_hypergraph.argmax(dim=1)

           hypergraph_train_acc = (
                pred_hypergraph == y_train_tensor
            ).float().mean().item()

        # ----------------------------------------------------
        # VALIDATION
        # NO VALIDATION PATIENT ENTERS HYPERGRAPH
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            logits_val, _, _ = model(
                xs_val,
                graph_mode=False
            )

            pred_val = logits_val.argmax(dim=1)

            val_acc = (
                pred_val == y_val_tensor
            ).float().mean().item()

        scheduler.step(val_acc)

        if epoch % 5 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"hypergraph Acc: {hypergraph_train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            counter = 0

        else:

            counter += 1

        if counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # --------------------------------------------------------
    # RESTORE BEST VALIDATION MODEL
    # --------------------------------------------------------

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    # --------------------------------------------------------
    # TRAIN / VAL / TEST EMBEDDINGS
    #
    # Z is always generated from the inductive adapter.
    # Therefore the downstream representation pathway is the
    # same for train, validation and test.
    # --------------------------------------------------------

    with torch.no_grad():

        _, _, Z_train = model(
            xs_train,
            graph_mode=True
        )

        _, _, Z_val = model(
            xs_val,
            graph_mode=False
        )

        _, _, Z_test = model(
            xs_test,
            graph_mode=False
        )

    Z_train = Z_train.detach().cpu().numpy()
    Z_val = Z_val.detach().cpu().numpy()
    Z_test = Z_test.detach().cpu().numpy()

    # --------------------------------------------------------
    # CLINICAL FUSION
    # --------------------------------------------------------

    X_fused_train = np.concatenate(
        [Z_train, X_cli_train],
        axis=1
    )

    X_fused_val = np.concatenate(
        [Z_val, X_cli_val],
        axis=1
    )

    X_fused_test = np.concatenate(
        [Z_test, X_cli_test],
        axis=1
    )

    # --------------------------------------------------------
    # RF SCALER
    # TRAIN ONLY
    # --------------------------------------------------------

    rf_scaler = StandardScaler()

    X_rf_train = rf_scaler.fit_transform(
        X_fused_train
    )

    X_rf_val = rf_scaler.transform(
        X_fused_val
    )

    X_rf_test = rf_scaler.transform(
        X_fused_test
    )

    # --------------------------------------------------------
    # VALIDATION THRESHOLD MODEL
    # TRAIN ONLY
    # --------------------------------------------------------

    rf_threshold = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf_threshold.fit(
        X_rf_train,
        y[train_idx]
    )

    val_prob = rf_threshold.predict_proba(
        X_rf_val
    )[:, 1]

    # --------------------------------------------------------
    # THRESHOLD SELECTED ONLY FROM VALIDATION
    # --------------------------------------------------------

    thresholds = np.arange(
        0.20,
        0.81,
        0.01
    )

    val_accuracy_list = []

    for threshold in thresholds:

        val_pred_tmp = (
            val_prob >= threshold
        ).astype(int)

        val_accuracy_list.append(
            accuracy_score(
                y[val_idx],
                val_pred_tmp
            )
        )

    best_threshold = float(
        thresholds[
            np.argmax(val_accuracy_list)
        ]
    )

    print(
        f"Validation threshold: {best_threshold:.2f}"
    )

    # --------------------------------------------------------
    # FINAL RF TRAINING
    # OUTER TEST IS NEVER USED
    # --------------------------------------------------------

    X_rf_trainval = np.concatenate(
        [
            X_fused_train,
            X_fused_val
        ],
        axis=0
    )

    y_rf_trainval = np.concatenate(
        [
            y[train_idx],
            y[val_idx]
        ],
        axis=0
    )

    rf_scaler_final = StandardScaler()

    X_rf_trainval = rf_scaler_final.fit_transform(
        X_rf_trainval
    )

    X_rf_test_final = rf_scaler_final.transform(
        X_fused_test
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_rf_trainval,
        y_rf_trainval
    )

    # --------------------------------------------------------
    # OUTER TEST PREDICTION
    # --------------------------------------------------------

    prob_test = rf.predict_proba(
        X_rf_test_final
    )[:, 1]

    pred_test = (
        prob_test >= best_threshold
    ).astype(int)

    y_test_true = y[test_idx]

    # --------------------------------------------------------
    # TEST METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test_true,
        pred_test
    )

    precision = precision_score(
        y_test_true,
        pred_test,
        zero_division=0
    )

    sensitivity = recall_score(
        y_test_true,
        pred_test,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test_true,
        pred_test,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test_true,
        pred_test,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test_true,
        pred_test
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_true,
        pred_test
    )

    cm = confusion_matrix(
        y_test_true,
        pred_test,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    if len(np.unique(y_test_true)) == 2:
        roc_auc = roc_auc_score(
            y_test_true,
            prob_test
        )
    else:
        roc_auc = np.nan

    fold_result = {
        "Fold": fold,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "ROC_AUC": roc_auc,
        "Precision": precision,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "MCC": mcc,
        "Balanced_Accuracy": balanced_accuracy,
        "Decision_Threshold": best_threshold,
        "Best_Val_Accuracy": best_val_acc
    }

    fold_metrics.append(fold_result)

    print("\n")
    print("-" * 80)
    print(f"FOLD {fold} TEST RESULTS")
    print("-" * 80)

    for key, value in fold_result.items():
        if key != "Fold":
            print(f"{key:22s}: {value:.4f}")

    # --------------------------------------------------------
    # SAVE OUT-OF-FOLD TEST PREDICTIONS
    # --------------------------------------------------------

    fold_test_results = pd.DataFrame({
        "Fold": fold,
        "Patient_ID": clinical_ids[test_idx],
        "True_Label": y_test_true,
        "Predicted_Label": pred_test,
        "Probability_Class_1": prob_test
    })

    all_test_predictions.append(
        fold_test_results
    )


# ============================================================
# S-9
# 10-FOLD CROSS-VALIDATION RESULTS
# ============================================================

fold_metrics_df = pd.DataFrame(
    fold_metrics
)

print("\n")
print("=" * 80)
print("10-FOLD CROSS-VALIDATION RESULTS")
print("=" * 80)

print(
    fold_metrics_df.to_string(
        index=False
    )
)

metric_columns = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "ROC_AUC",
    "Precision",
    "Macro_F1",
    "Weighted_F1",
    "MCC",
    "Balanced_Accuracy"
]

cv_summary = pd.DataFrame({
    "Metric": metric_columns,
    "Mean": [
        fold_metrics_df[m].mean()
        for m in metric_columns
    ],
    "Std": [
        fold_metrics_df[m].std()
        for m in metric_columns
    ]
})

print("\n")
print("=" * 80)
print("10-FOLD MEAN ± STD")
print("=" * 80)

for _, row in cv_summary.iterrows():

    print(
        f"{row['Metric']:22s}: "
        f"{row['Mean']:.4f} ± {row['Std']:.4f}"
    )


# ============================================================
# S-10
# OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_results = pd.concat(
    all_test_predictions,
    axis=0,
    ignore_index=True
)

oof_results = oof_results.sort_values(
    "Patient_ID"
).reset_index(drop=True)

print("\n")
print("=" * 80)
print("OUT-OF-FOLD TEST PREDICTIONS")
print("=" * 80)

print(
    oof_results.head()
)

print(
    "\nNumber of OOF predictions:",
    len(oof_results)
)

assert len(oof_results) == len(y)

assert (
    oof_results["Patient_ID"].nunique()
    == len(y)
)


# ============================================================
# S-11
# AGGREGATED 10-FOLD OUT-OF-FOLD METRICS
# ============================================================

y_oof = oof_results[
    "True_Label"
].values

pred_oof = oof_results[
    "Predicted_Label"
].values

prob_oof = oof_results[
    "Probability_Class_1"
].values

oof_accuracy = accuracy_score(
    y_oof,
    pred_oof
)

oof_precision = precision_score(
    y_oof,
    pred_oof,
    zero_division=0
)

oof_sensitivity = recall_score(
    y_oof,
    pred_oof,
    zero_division=0
)

oof_macro_f1 = f1_score(
    y_oof,
    pred_oof,
    average="macro",
    zero_division=0
)

oof_weighted_f1 = f1_score(
    y_oof,
    pred_oof,
    average="weighted",
    zero_division=0
)

oof_mcc = matthews_corrcoef(
    y_oof,
    pred_oof
)

oof_balanced_accuracy = balanced_accuracy_score(
    y_oof,
    pred_oof
)

oof_cm = confusion_matrix(
    y_oof,
    pred_oof,
    labels=[0, 1]
)

oof_tn, oof_fp, oof_fn, oof_tp = oof_cm.ravel()

oof_specificity = (
    oof_tn / (oof_tn + oof_fp)
    if (oof_tn + oof_fp) > 0
    else 0.0
)

oof_roc_auc = roc_auc_score(
    y_oof,
    prob_oof
)

print("\n")
print("=" * 80)
print("AGGREGATED OUT-OF-FOLD RESULTS")
print("=" * 80)

print(
    f"Accuracy            : {oof_accuracy:.4f}"
)
print(
    f"Sensitivity         : {oof_sensitivity:.4f}"
)
print(
    f"Specificity         : {oof_specificity:.4f}"
)
print(
    f"ROC-AUC             : {oof_roc_auc:.4f}"
)
print(
    f"Precision           : {oof_precision:.4f}"
)
print(
    f"Macro_F1            : {oof_macro_f1:.4f}"
)
print(
    f"Weighted_F1         : {oof_weighted_f1:.4f}"
)
print(
    f"MCC                 : {oof_mcc:.4f}"
)
print(
    f"Balanced_Accuracy   : {oof_balanced_accuracy:.4f}"
)


# ============================================================
# S-12
# SAVE 10-FOLD RESULTS
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "HATS_Inductive_10Fold_Results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

fold_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10fold_fold_metrics.csv"
    ),
    index=False
)

cv_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10fold_mean_std_metrics.csv"
    ),
    index=False
)

oof_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10fold_out_of_fold_predictions.csv"
    ),
    index=False
)

oof_metrics = pd.DataFrame([
    {
        "Accuracy": oof_accuracy,
        "Sensitivity": oof_sensitivity,
        "Specificity": oof_specificity,
        "ROC_AUC": oof_roc_auc,
        "Precision": oof_precision,
        "Macro_F1": oof_macro_f1,
        "Weighted_F1": oof_weighted_f1,
        "MCC": oof_mcc,
        "Balanced_Accuracy": oof_balanced_accuracy
    }
])

oof_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10fold_aggregated_oof_metrics.csv"
    ),
    index=False
)

print("\nSaved results to:")
print(OUTPUT_DIR)
print("10fold_fold_metrics.csv")
print("10fold_mean_std_metrics.csv")
print("10fold_out_of_fold_predictions.csv")
print("10fold_aggregated_oof_metrics.csv")
