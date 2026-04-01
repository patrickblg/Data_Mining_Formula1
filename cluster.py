import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Încărcarea datelor
df = pd.read_csv('data.csv')

# 2. Preprocesare (Filtrare pentru consistență cu modelul de regresie)
df['AvgPitStopTime'] = pd.to_numeric(df['AvgPitStopTime'], errors='coerce')
# Păstrăm doar opririle "normale" (15-60s) pentru a elimina accidentele sau erorile
df_clean = df[(df['AvgPitStopTime'] > 15) & (df['AvgPitStopTime'] < 60)].dropna(subset=['AvgPitStopTime'])

# --- ANALIZĂ PRELIMINARĂ ---
plt.figure(figsize=(15, 5))

# Distribuția targetului
plt.subplot(1, 2, 1)
sns.histplot(df_clean['AvgPitStopTime'], kde=True, color='royalblue')
plt.title('Distribuția Timpului de Pit Stop (Target)')

# Relația între Temperatura Pistei și Timpul de oprire
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_clean, x='Track_Temp_C', y='AvgPitStopTime', alpha=0.4)
plt.title('Impactul Temperaturii Pistei asupra Pit Stop-ului')
plt.savefig('preliminary_analysis.png')
plt.close()

# --- CLUSTERING (Identificarea Pattern-urilor) ---
# Selectăm variabilele care definesc contextul unei opriri
features_clustering = ['AvgPitStopTime', 'Track_Temp_C', 'Laps', 'TotalPitStops', 'Position']
X = df_clean[features_clustering].fillna(df_clean[features_clustering].median())

# Standardizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicăm K-Means (3 clustere pentru: Rapid, Standard, Atipic)
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Vizualizarea clusterelor folosind PCA (reducerea dimensiunii pentru grafic 2D)
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