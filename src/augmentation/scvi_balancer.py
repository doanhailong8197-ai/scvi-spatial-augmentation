import math
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from anndata import AnnData
import scvi


def augment_data(
    adata: AnnData,
    labels_key: str = "cell_type",
    covariate_batch_key: str = "batch",
    layer_key: str = "counts",
    target_count: int = 2223,
    max_epochs: int = 500,
    n_latent: int = 20,
    n_hidden: int = 128,
    n_layers: int = 2,
    dropout_rate: float = 0.1
) -> AnnData:
    """
    Generate synthetic cells using scVI to balance spatial/RNA transcriptomics datasets.
    The generative process respects the l_n scaling factor (correlated with log-library size) 
    and isolates the experimental batch notation from the generative latent variables z.
    """
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=layer_key,
        batch_key=covariate_batch_key,
        labels_key=labels_key
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )

    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    print(f"Hardware accelerator: {accelerator.upper()}")

    model.train(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices="auto",
        batch_size=1024,
        early_stopping=True,
        early_stopping_patience=10,
    )

    aug_X_list = []
    aug_obs_list = []
    cell_types = adata.obs[labels_key].unique()

    print("Initiating generative data augmentation based on learned l_n scaling factors...")

    for ct in cell_types:
        indices = np.where(adata.obs[labels_key] == ct)[0]
        n_current = len(indices)

        if n_current < target_count:
            n_missing = target_count - n_current
            n_samples_ratio = math.ceil(n_missing / n_current)

            print(f"Class '{ct}': Generating {n_missing} synthetic cells")

            pp = model.posterior_predictive_sample(
                adata=adata,
                indices=indices,
                n_samples=n_samples_ratio
            )

            if sp.issparse(pp):
                arr = pp.toarray()
            elif hasattr(pp, "to_numpy"):
                arr = pp.to_numpy()
            elif isinstance(pp, np.ndarray):
                arr = pp
            elif isinstance(pp, torch.Tensor):
                arr = pp.detach().cpu().numpy()
            elif hasattr(pp, "todense"):
                arr = pp.todense()
            else:
                arr = np.asarray(pp)

            if arr.ndim == 3:
                n_c, n_g, n_s = arr.shape
                arr_reshaped = arr.transpose(0, 2, 1).reshape(n_c * n_s, n_g)
            elif arr.ndim == 2:
                arr_reshaped = arr
            else:
                continue

            if arr_reshaped.shape[0] >= n_missing:
                arr_synthetic_dense = arr_reshaped[:n_missing, :]
            else:
                arr_synthetic_dense = arr_reshaped

            arr_synthetic_sparse = sp.csr_matrix(arr_synthetic_dense)
            aug_X_list.append(arr_synthetic_sparse)

            obs_synthetic = pd.DataFrame({
                labels_key: [ct] * arr_synthetic_sparse.shape[0],
                "source": ["scvi_augmented"] * arr_synthetic_sparse.shape[0],
                "is_synthetic": [True] * arr_synthetic_sparse.shape[0]
            })
            aug_obs_list.append(obs_synthetic)

    if len(aug_X_list) > 0:
        final_aug_X = sp.vstack(aug_X_list)
        final_aug_obs = pd.concat(aug_obs_list, axis=0).reset_index(drop=True)
        final_aug_obs.index = [f"aug_cell_{i}" for i in range(len(final_aug_obs))]
    else:
        final_aug_X = sp.csr_matrix((0, adata.n_vars))
        final_aug_obs = pd.DataFrame(columns=[labels_key, "source", "is_synthetic"])

    if sp.issparse(final_aug_X):
        final_aug_X.data = np.rint(final_aug_X.data).astype(np.float32)
    else:
        final_aug_X = np.rint(final_aug_X).astype(np.float32)

    adata_aug = AnnData(X=final_aug_X)
    adata_aug.obs = final_aug_obs.copy()
    adata_aug.var = adata.var.copy()
    adata_aug.var_names = adata.var_names.copy()

    adata.obs["source"] = "original"
    adata.obs["is_synthetic"] = False

    print("Concatenating original and synthetic datasets...")
    
    adata_combined = ad.concat(
        [adata, adata_aug],
        label="augmentation_batch",
        keys=["original", "augmented"],
        index_unique="-"
    )

    print("Final balanced cell type distribution:")
    print(pd.crosstab(adata_combined.obs[labels_key], adata_combined.obs["augmentation_batch"], margins=True))

    return adata_combined