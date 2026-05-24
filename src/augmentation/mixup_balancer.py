import gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

def augment_data(
    adata: AnnData, 
    label_col: str = 'cell_type', 
    alpha: float = 0.4, 
    random_state: int = 42
) -> AnnData:
    rng = np.random.default_rng(random_state)
    
    counts = adata.obs[label_col].value_counts()
    target_count = counts.max()
    print(f"Target count per class: {target_count}")

    initial_X = adata.X.copy()
    if sp.issparse(initial_X) and not sp.isspmatrix_csr(initial_X):
        initial_X = initial_X.tocsr()

    X_list = [initial_X]
    obs_list = [adata.obs.copy()]
    all_indices = np.arange(adata.n_obs)

    for cell_type, count in counts.items():
        n_needed = target_count - count
        if n_needed <= 0:
            continue

        print(f"Augmenting class '{cell_type}': generating {n_needed} synthetic cells")

        mask = (adata.obs[label_col] == cell_type).values
        class_indices = np.where(mask)[0]

        idx1 = rng.choice(class_indices, size=n_needed)
        idx2 = rng.choice(all_indices, size=n_needed)

        X1 = adata.X[idx1]
        X2 = adata.X[idx2]

        if not sp.isspmatrix_csr(X1):
            X1 = X1.tocsr()
        if not sp.isspmatrix_csr(X2):
            X2 = X2.tocsr()

        lam = rng.beta(alpha, alpha, size=n_needed)
        lam_diag = sp.diags(lam)
        inv_lam_diag = sp.diags(1 - lam)

        X_aug = lam_diag.dot(X1) + inv_lam_diag.dot(X2)

        X_aug.data = np.clip(X_aug.data, 0, None)
        X_aug.data[X_aug.data < 1e-4] = 0
        X_aug.eliminate_zeros()
        
        new_obs = adata.obs.iloc[idx1].copy()
        new_obs.index = [f"mix_{i}_{cell_type}" for i in range(n_needed)]
        
        y1 = adata.obs.iloc[idx1][label_col].values
        y2 = adata.obs.iloc[idx2][label_col].values
        choose_y1 = rng.random(n_needed) < lam
        new_labels = np.where(choose_y1, y1, y2)
        
        new_obs[label_col] = new_labels

        X_list.append(X_aug)
        obs_list.append(new_obs)

        del X1, X2, X_aug, lam_diag, inv_lam_diag
        gc.collect()

    print("Concatenating augmented data matrices...")
    final_X = sp.vstack(X_list)
    final_obs = pd.concat(obs_list)

    del X_list, obs_list
    gc.collect()

    final_adata = sc.AnnData(X=final_X, obs=final_obs, var=adata.var.copy())

    return final_adata