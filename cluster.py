import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
import numpy as np

# 1. Încărcarea datelor
df = pd.read_csv('data.csv')

# 2. Preprocesare
df['AvgPitStopTime'] = pd.to_numeric(df['AvgPitStopTime'], errors='coerce')
df_clean = df[(df['AvgPitStopTime'] > 15) & (df['AvgPitStopTime'] < 60)].dropna(subset=['AvgPitStopTime'])

# --- ANALIZĂ PRELIMINARĂ ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_clean['AvgPitStopTime'], kde=True, color='royalblue')
plt.title('Distribuția Timpului de Pit Stop (Target)')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_clean, x='Track_Temp_C', y='AvgPitStopTime', alpha=0.4)
plt.title('Impactul Temperaturii Pistei asupra Pit Stop-ului')
plt.savefig('preliminary_analysis.png')
plt.close()

# --- CLUSTERING ---
features_clustering = ['AvgPitStopTime', 'Track_Temp_C', 'Laps', 'TotalPitStops', 'Position']
X_clust = df_clean[features_clustering].fillna(df_clean[features_clustering].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df_clean['Cluster'].values

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis')
plt.title('Segmentarea Datelor prin Clustering (K-Means)')
plt.savefig('clustering_visualization.png')
plt.close()

print("Analiza preliminară finalizată.")

# ==============================================================================
# --- PREDICȚIE: Random Forest ---
# ==============================================================================

features = ['Track_Temp_C', 'Laps', 'TotalPitStops', 'Position', 'Cluster']
target = 'AvgPitStopTime'

X = df_clean[features].fillna(df_clean[features].median())
y = df_clean[target]

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenare
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluare
y_pred = model.predict(X_test)
print(f"\nMAE  : {mean_absolute_error(y_test, y_pred):.3f}s")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}s")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")

# Grafic: Predicții vs Valori Reale
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valori Reale (s)')
plt.ylabel('Valori Prezise (s)')
plt.title('Random Forest — Predicții vs Valori Reale')
plt.tight_layout()
plt.savefig('model_evaluation.png')
plt.close()

print("\nGrafic salvat: model_evaluation.png")
print("Finalizat Regression.")

#
#Podium prediction Classification
#
print("\n" + "=" * 60)
print("  ANALIZĂ TEMPORALĂ: TRAIN (2018–2023) → TEST (2024)")
print("=" * 60)

# Target: podium
df_clean['OnPodium'] = (df_clean['Position'] <= 3).astype(int)

# -----------------------------
# 1. Split TEMPORAL (NU random)
# -----------------------------

train_df = df_clean[df_clean['Season'] <= 2023]
test_df  = df_clean[df_clean['Season'] == 2024]

print(f"Train samples: {len(train_df)}")
print(f"Test samples (2024): {len(test_df)}")

# -----------------------------
# 2. Feature-uri SAFE
# -----------------------------

features_cls = [
    'Track_Temp_C',
    'AvgPitStopTime',
    'Driver Aggression Score'
]

X_train = train_df[features_cls].fillna(train_df[features_cls].median())
y_train = train_df['OnPodium']

X_test  = test_df[features_cls].fillna(train_df[features_cls].median())
y_test  = test_df['OnPodium']

# -----------------------------
# 3. Antrenare model
# -----------------------------

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

clf.fit(X_train, y_train)

# -----------------------------
# 4. Predicții 2024
# -----------------------------

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

test_df = test_df.copy()
test_df['Predicted_OnPodium'] = y_pred
test_df['Podium_Probability'] = y_prob

# -----------------------------
# 5. Evaluare REALĂ pe 2024
# -----------------------------

print("\n--- Raport Clasificare (2024) ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Nu Podium', 'Podium']
))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Nu Podium', 'Podium'],
    yticklabels=['Nu Podium', 'Podium']
)
plt.xlabel('Prezis')
plt.ylabel('Real')
plt.title('Confusion Matrix — Predicții 2024')
plt.tight_layout()
plt.savefig('podium_confusion_matrix_2024.png')
plt.close()

print("Grafic salvat: podium_confusion_matrix_2024.png")

# -----------------------------
# 6. Acuratețe simplă
# -----------------------------

accuracy_2024 = (y_pred == y_test).mean()
print(f"\nAcuratețe predicții podium 2024: {accuracy_2024:.2%}")
