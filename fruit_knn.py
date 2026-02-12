import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Class names: 0=Apple, 1=Orange, 2=Banana
CLASS_NAMES = ["Apple", "Orange", "Banana"]

fruits = np.array([
    [150, 7],   # Apple
    [170, 6],   # Apple
    [140, 8],   # Apple
    [165, 7],   # Apple
    [180, 4],   # Orange
    [190, 5],   # Orange
    [175, 3],   # Orange
    [200, 4],   # Orange
    [120, 9],   # Banana 
    [130, 8],   # Banana 
    [125, 9],   # Banana 
])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])  # 0=Apple, 1=Orange, 2=Banana

mystery_fruits = np.array([
    [160, 7],   # mystery 1
    [185, 4],   # mystery 2
    [125, 8],   # mystery 3
])

K_VALUES = [1, 3, 5]

# ========== 1. VISUALIZE DATA ==========
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.scatter(fruits[labels == 0, 0], fruits[labels == 0, 1], c='green', s=120, label='Apple', edgecolors='black')
ax1.scatter(fruits[labels == 1, 0], fruits[labels == 1, 1], c='orange', s=120, label='Orange', edgecolors='black')
ax1.scatter(fruits[labels == 2, 0], fruits[labels == 2, 1], c='gold', s=120, label='Banana', edgecolors='black')
ax1.scatter(mystery_fruits[:, 0], mystery_fruits[:, 1], c='red', s=200, marker='*', label='Mystery', edgecolors='black', zorder=5)
ax1.set_xlabel('Weight (grams)')
ax1.set_ylabel('Sweetness (1-10)')
ax1.set_title('Fruit Data')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('/Users/macbookm1/Documents/CSML/fruit_knn_plot.png', dpi=100)

def draw_neighbor_plot(axes, k, knn, mystery_fruits, fruits, labels, class_names):
    """Draw 3 subplots (M1, M2, M3) for given k, showing neighbors for each mystery fruit."""
    for col, (mf, ax) in enumerate(zip(mystery_fruits, axes)):
        mf_2d = mf.reshape(1, -1)
        pred = knn.predict(mf_2d)[0]
        dists, inds = knn.kneighbors(mf_2d)
        ax.scatter(fruits[labels == 0, 0], fruits[labels == 0, 1], c='green', s=80, alpha=0.5, label='Apple')
        ax.scatter(fruits[labels == 1, 0], fruits[labels == 1, 1], c='orange', s=80, alpha=0.5, label='Orange')
        ax.scatter(fruits[labels == 2, 0], fruits[labels == 2, 1], c='gold', s=80, alpha=0.5, label='Banana')
        ax.scatter(mystery_fruits[:, 0], mystery_fruits[:, 1], c='red', s=120, marker='*', alpha=0.4, zorder=2)
        ax.scatter(mf[0], mf[1], c='red', s=220, marker='*', edgecolors='black', zorder=5, label='Mystery' if col == 0 else None)
        for idx in inds[0]:
            ax.plot([mf[0], fruits[idx, 0]], [mf[1], fruits[idx, 1]], 'b--', alpha=0.6)
        ax.scatter(fruits[inds[0], 0], fruits[inds[0], 1], c='blue', s=140, marker='o', edgecolors='black', linewidths=1.5, label='Neighbors' if col == 0 else None)
        ax.set_xlabel('Weight (g)')
        ax.set_ylabel('Sweetness')
        ax.set_title(f'M{col + 1} → {class_names[pred]}')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc='upper left', fontsize=8)
    return axes

# ========== 2. THREE PLOTS: k=1, k=3, k=5 (each with M1, M2, M3) ==========
for k in K_VALUES:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'k={k}', fontsize=14)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(fruits, labels)
    draw_neighbor_plot(axes, k, knn, mystery_fruits, fruits, labels, CLASS_NAMES)
    plt.tight_layout()
    plt.savefig(f'/Users/macbookm1/Documents/CSML/fruit_knn_plot_k{k}.png', dpi=100)
    plt.show()

for k in K_VALUES:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(fruits, labels)

    print("=" * 55)
    print(f"KNN with k = {k}")
    print("=" * 55)

    for m_idx, mf in enumerate(mystery_fruits, 1):
        mf_2d = mf.reshape(1, -1)
        prediction = knn.predict(mf_2d)[0]
        pred_label = CLASS_NAMES[prediction]
        distances, indices = knn.kneighbors(mf_2d)
        neighbor_labels = labels[indices[0]]
        neighbor_votes = [CLASS_NAMES[l] for l in neighbor_labels]

        print(f"\n  Mystery fruit #{m_idx} [{mf[0]}g, {mf[1]} sweetness] → Predicted: {pred_label} (class {prediction})")
        print("  Neighbors that voted:")
        for i, (idx, dist, vote) in enumerate(zip(indices[0], distances[0], neighbor_votes), 1):
            w, s = fruits[idx]
            print(f"    #{i}: [{w}g, {s}] → {vote} (distance: {dist:.2f})")
        print(f"  Majority: {pred_label} ({np.sum(neighbor_labels == prediction)}/{k} neighbors)")
    print()
