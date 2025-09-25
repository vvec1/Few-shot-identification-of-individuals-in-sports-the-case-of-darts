import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import genpareto

from functions import preprocess, create_LA_horizonta_LA_vertical, create_zeroed_QTM
from loader import load_data
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd
import sys
sys.path.append(r"C:\Users\valve\LKN Dropbox\Val Vec\meritve darts\clean measurements\MORE PEOPLE\TS2Vec")


from ts2vec import TS2Vec


# --- Plotting Embeddings ---
def plot_embeddings(embeddings, labels, title="Embeddings", method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    emb_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette='tab10', s=60)
    plt.title(f"{title} ({method.upper()})")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Prototypical Network Utils ---
def sample_episode(X, y, N=3, K=5, Q=5):
    classes = np.random.choice(np.unique(y), N, replace=False)
    support_x, support_y, query_x, query_y = [], [], [], []
    for i, c in enumerate(classes):
        idx = np.where(y == c)[0]
        sel = np.random.choice(idx, K + Q, replace=False)
        support_x.append(X[sel[:K]])
        query_x.append(X[sel[K:]])
        support_y += [i] * K
        query_y += [i] * Q
    return (np.concatenate(support_x), np.array(support_y),
            np.concatenate(query_x), np.array(query_y))

def proto_logits(model, support_x, support_y, query_x, device):
    def encode_ts(batch_x):
        # batch_x is a numpy array. Pass it directly to model.encode
        with torch.no_grad():
            # model.encode takes a numpy array and returns a numpy array
            emb = model.encode(batch_x, encoding_window='full_series')
        
        # Convert the numpy output from encode() to a tensor for further processing
        emb_tensor = torch.from_numpy(emb).to(device)
        return nn.functional.normalize(emb_tensor, dim=1)

    emb_sup = encode_ts(support_x)
    emb_q = encode_ts(query_x)
    sy = torch.tensor(support_y, dtype=torch.long).to(device)

    N = len(torch.unique(sy))
    prototypes = torch.stack([emb_sup[sy == i].mean(0) for i in range(N)])

    dists = torch.cdist(emb_q, prototypes)
    return -dists

# --- Main ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load & preprocess data
    df, classes, df_list = load_data()
    for i in sorted([i for i,d in enumerate(df_list) if d.empty], reverse=True):
        df_list.pop(i); classes.pop(i)
    for d in df_list:
        for col in ['empty3','LV Ts','SAMPLE ID','QTM TIME1','Current event']:
            if col in d: d.drop(columns=[col], inplace=True)
    create_zeroed_QTM(df_list, per_throw=False)

    START, END = 0, 100
    feats = []
    for d in df_list:
        f = create_LA_horizonta_LA_vertical(d)
        f = preprocess(f)[['LA_hor','LA_vert']]
        feats.append(f.values[START:END])
    X = np.stack(feats)
    y = np.array(classes)

    # seen / unseen / open-set classes
    all_cls = sorted(set(y))
    unseen = all_cls[10:18]
    unseen_os = all_cls[18:20]
    excluded = unseen + unseen_os
    seen_mask = ~np.isin(y, excluded)
    X_seen, y_seen = X[seen_mask], y[seen_mask]
    X_unseen, y_unseen = X[np.isin(y, unseen)], y[np.isin(y, unseen)]
    X_unseen_os, y_unseen_os = X[np.isin(y, unseen_os)], y[np.isin(y, unseen_os)]

    X_tr, X_test, y_tr, y_test = train_test_split(
        X_seen, y_seen, test_size=0.3, stratify=y_seen, random_state=42
    )

    # Load pretrained TS2Vec model
    model = TS2Vec(input_dims=2, output_dims=60, device=device)
    model.fit(X_tr,  verbose=True)
    
    



    '''
    # Compute embeddings for support + query
    sx_tensor = torch.tensor(sx_test, dtype=torch.float32).permute(0,2,1).to(device)
    qx_tensor = torch.tensor(qx_test, dtype=torch.float32).permute(0,2,1).to(device)
    emb_s = model(sx_tensor).cpu().detach().numpy()
    emb_q = model(qx_tensor).cpu().detach().numpy()
    all_emb = np.concatenate([emb_s, emb_q])
    all_labels = np.concatenate([sy_test, qy_test])
    plot_embeddings(all_emb, all_labels, title="Seen Test Embeddings", method="umap")
    plot_embeddings(all_emb, all_labels, title="Seen Test Embeddings", method="tsne")
    '''

    # --- Episodic eval on unseen ---
    n_ep, n_way, n_shot, n_q = 100, 5, 5, 15
    accs = []
    all_true_un, all_pred_un = [], []
    for _ in range(n_ep):
        sx, sy, qx, qy = sample_episode(X_unseen, y_unseen, N=n_way, K=n_shot, Q=n_q)
        logits = proto_logits(model, sx, sy, qx, device)
        preds = logits.argmax(1).cpu().numpy()
        accs.append((preds == qy).mean())
        all_true_un.extend(qy)
        all_pred_un.extend(preds)


        qx_tensor_un = torch.tensor(qx, dtype=torch.float32).permute(0,2,1).to(device)
        emb_q_unseen = model.encode(qx, encoding_window='full_series')
       # plot_embeddings(emb_q_unseen, qy, title="Unseen Episode Embeddings", method="tsne")
    
    print(f"Unseen Episodic Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Confusion matrix for unseen classes
    cm_un = confusion_matrix(all_true_un, all_pred_un, labels=list(range(n_way)))
    plt.figure()
    sns.heatmap(cm_un, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(range(n_way)), yticklabels=list(range(n_way)))
    plt.title("Confusion Matrix: Unseen Episodic")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()




# --- Collect embeddings for unseen and unseen_os ---
def get_embeddings(model, X, y, device):
#    model.eval()
    with torch.no_grad():
        # x = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device) # Remove this line
        # emb = model(x).cpu().numpy() # Change this line
        emb = model.encode(X, encoding_window='full_series') # New line
    return emb, y

# Get embeddings
emb_unseen, y_unseen_ids = get_embeddings(model, X_unseen, y_unseen, device)
emb_os, y_os_ids = get_embeddings(model, X_unseen_os, y_unseen_os, device)

# Combine
emb_all = np.concatenate([emb_unseen, emb_os])
labels_all = np.concatenate([y_unseen_ids, y_os_ids])
label_type = ['unseen'] * len(y_unseen_ids) + ['unseen_os'] * len(y_os_ids)

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
emb_2d = tsne.fit_transform(emb_all)

# Plot with seaborn
df_plot = pd.DataFrame({
    'x': emb_2d[:, 0],
    'y': emb_2d[:, 1],
    'Class': labels_all,
    'Type': label_type
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_plot, x='x', y='y', hue='Class', style='Type', palette='tab10', s=60)
plt.title("Unseen & Open-Set Embeddings (t-SNE)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

umap_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
emb_2d_umap = umap_reducer.fit_transform(emb_all)

# Create dataframe for plotting
df_plot_umap = pd.DataFrame({
    'x': emb_2d_umap[:, 0],
    'y': emb_2d_umap[:, 1],
    'Class': labels_all,
    'Type': label_type
})

# Plot with seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_plot_umap, x='x', y='y', hue='Class', style='Type', palette='tab10', s=60)
plt.title("Unseen & Open-Set Embeddings (UMAP)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





seen_embeddings, seen_labels = get_embeddings(model, X_seen, y_seen, device)

emb_all = np.vstack([
    seen_embeddings,
    emb_unseen,
    emb_os
])

# Concatenate all class labels
labels_all = np.concatenate([
    seen_labels,
    y_unseen_ids,
    np.full(len(emb_os), -1)  # Use -1 for unknown class
])

# Create label types
label_type = (
    ['seen'] * len(seen_embeddings) +
    ['unseen'] * len(y_unseen_ids) +
    ['unseen_os'] * len(emb_os)
)





# UMAP reduction
umap_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
emb_2d_umap = umap_reducer.fit_transform(emb_all)  # emb_all includes seen, unseen, and unseen_os

# DataFrame for plotting
df_plot_umap = pd.DataFrame({
    'x': emb_2d_umap[:, 0],
    'y': emb_2d_umap[:, 1],
    'Class': labels_all,    # e.g., 0–9 for seen, 10–19 for unseen
    'Type': label_type      # "seen", "unseen", "unseen_os"
})

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_plot_umap,
    x='x', y='y',
    hue='Class',
    style='Type',
    palette='tab20',
    s=60
)

plt.title("Seen, Unseen & Open-Set Embeddings (UMAP)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



from sklearn.metrics import pairwise_distances

def compute_intra_inter_detailed(embeddings, labels):
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    class_to_embeddings = {cls: embeddings[labels == cls] for cls in unique_classes}
    class_indices = {cls: i for i, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    # Intra-class distances
    intra_class_dists = np.zeros(num_classes)
    for cls, embs in class_to_embeddings.items():
        i = class_indices[cls]
        if len(embs) < 2:
            intra_class_dists[i] = np.nan  # or 0, depending on preference
        else:
            dists = pairwise_distances(embs)
            intra = dists[np.triu_indices_from(dists, k=1)]
            intra_class_dists[i] = intra.mean()

    # Inter-class distances (centroid to centroid)
    centroids = [embs.mean(axis=0) for embs in class_to_embeddings.values()]
    centroid_matrix = pairwise_distances(centroids)
    inter_class_dists = centroid_matrix  # shape (num_classes, num_classes)

    return unique_classes, intra_class_dists, inter_class_dists


classes, intra_dists, inter_dists = compute_intra_inter_detailed(emb_all, labels_all)

# Analysis example:
import matplotlib.pyplot as plt
import seaborn as sns

# Intra-class per class
plt.figure(figsize=(8,4))
sns.barplot(x=classes, y=intra_dists)
plt.title("Intra-class Distance per Seen Class")
plt.ylabel("Mean Intra-class Distance")
plt.xlabel("Class")
plt.show()

# Mask the diagonal (self-distance)
inter_dists_no_diag = inter_dists.copy()
np.fill_diagonal(inter_dists_no_diag, np.inf)

# Plot
plt.figure(figsize=(8, 4))
sns.barplot(x=classes, y=inter_dists_no_diag.min(axis=1))
plt.title("Minimum Inter-Class Distance per Seen Class (Excl. Self)")
plt.ylabel("Minimum Inter-Class Distance")
plt.xlabel("Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("inter class distances")
print("Seen average", inter_dists.mean(axis=1)[1:10].mean())
print("Unseen average", inter_dists.mean(axis=1)[10:].mean())
print("___________________________________________")
seen_class_indices = np.array([i for i in range (10)])
unseen_class_indices = np.array([i for i in range(10, 17)])


seen_intra = intra_dists[seen_class_indices]
unseen_intra = intra_dists[unseen_class_indices]

print("Avg Intra-class Distance (Seen):", seen_intra.mean())
print("Avg Intra-class Distance (Unseen):", unseen_intra.mean())


def prototype_distance_variance(support_embs):
    proto = support_embs.mean(axis=0)
    dists = np.linalg.norm(support_embs - proto, axis=1)
    return np.var(dists)

from collections import defaultdict

class_to_indices = defaultdict(list)
for idx, label in enumerate(labels_all):
    class_to_indices[label].append(idx)

# Convert to list of arrays for each class
    
emb_seen =[]
emb_unseen = []
labels_seen = []
labels_unseen = []

seen_proto_var = []

for cls in seen_class_indices:
    embs = emb_all[class_to_indices[cls]]  # shape: [n_samples, feature_dim]
    
    emb_seen.extend(embs)
    labels_seen.extend(np.full(len(embs), cls))

    proto = embs.mean(axis=0)
    dists = np.linalg.norm(embs - proto, axis=1)
    seen_proto_var.append(np.var(dists))

unseen_proto_var = []
for cls in unseen_class_indices:
    embs = emb_all[class_to_indices[cls]]

    emb_unseen.extend(embs)
    labels_unseen.extend(np.full(len(embs), cls))

    proto = embs.mean(axis=0)
    dists = np.linalg.norm(embs - proto, axis=1)
    unseen_proto_var.append(np.var(dists))

print("Avg Prototype Variance (Seen):", np.mean(seen_proto_var))
print("Avg Prototype Variance (Unseen):", np.mean(unseen_proto_var))

from sklearn.metrics import davies_bouldin_score

dbi_seen = davies_bouldin_score(emb_seen, labels_seen)
dbi_unseen = davies_bouldin_score(emb_unseen, labels_unseen)

print("Davies-Bouldin Index (Seen):", dbi_seen) 
print("Davies-Bouldin Index (Unseen):", dbi_unseen)