from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def plot_standard_confusion_matrix(y_true, y_pred, filename=None):
    """Draws a standard confusion matrix for All-Known tasks."""
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)

    ax.set_title('Standard Confusion Matrix', pad=20)
    ax.set_xlabel('Predicted Label', labelpad=10)
    ax.set_ylabel('Ground Truth', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if filename:
        save_figure(fig, filename, format='pdf')
    plt.show()

def plot_novel_discovery_confusion_matrix(y_true, y_pred, known_types, filename=None):
    """Draws a structured confusion matrix separating Known and Novel cells."""
    all_true_labels = np.unique(y_true)
    novel_types = sorted([t for t in all_true_labels if t not in known_types])

    display_labels_true = list(known_types) + novel_types
    display_labels_pred = list(known_types) + ['Unknown']

    cm_df = pd.crosstab(
        pd.Series(y_true, name='Ground Truth'),
        pd.Series(y_pred, name='Prediction')
    )
    cm_df = cm_df.reindex(index=display_labels_true, columns=display_labels_pred, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5, cbar=False, ax=ax)

    ax.set_title('Confusion Matrix: Known Labels vs Novel Discovery', pad=20)
    ax.set_xlabel('Prediction', labelpad=10)
    ax.set_ylabel('Ground Truth', labelpad=10)

    # Separation line
    ax.axhline(y=len(known_types), color='red', linestyle='--', linewidth=2)
    
    # Boundary annotations
    ax.text(len(display_labels_pred) + 0.5, len(known_types) / 2, 'KNOWN CELLS',
             color='red', rotation=270, va='center', fontweight='bold')
    ax.text(len(display_labels_pred) + 0.5, len(known_types) + len(novel_types) / 2, 'NOVEL CELLS',
             color='red', rotation=270, va='center', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if filename:
        save_figure(fig, filename, format='pdf')
    plt.show()
    
def plot_celltype_distribution(adata_dict, label_col='cell_type', filename=None):
    """
    Plot cell type distributions for multiple datasets in a single comparative bar chart.
    
    Parameters:
    - adata_dict: dict of {Dataset_Name: AnnData_Object}
    - label_col: string, column name in obs containing cell types
    - filename: string, filename to save the figure
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df_list = []
    for name, adata in adata_dict.items():
        if label_col in adata.obs.columns:
            counts = adata.obs[label_col].value_counts().reset_index()
            counts.columns = ['Cell Type', 'Count']
            counts['Dataset'] = name
            df_list.append(counts)
            
    if not df_list:
        print(f"Warning: Label column '{label_col}' not found.")
        return

    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Sort order based on the first dataset's frequencies
    first_dataset = list(adata_dict.keys())[0]
    sort_order = df_combined[df_combined['Dataset'] == first_dataset].sort_values(by='Count', ascending=False)['Cell Type']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_combined, 
        x='Cell Type', 
        y='Count', 
        hue='Dataset', 
        order=sort_order,
        palette='muted',
        ax=ax
    )
    
    ax.set_title('Cell Type Distribution Across Datasets', pad=15)
    ax.set_xlabel('Cell Type', labelpad=10)
    ax.set_ylabel('Number of Cells', labelpad=10)
    
    plt.xticks(rotation=45, ha='right')
    ax.legend(frameon=False, title='Dataset')
    sns.despine() # Remove top and right borders
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename, format='pdf')
    plt.show()