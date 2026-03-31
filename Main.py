"""
=============================================================
  F1 Pit Stop Time Prediction — Regression
  Data Mining Project
=============================================================
  Target:   AvgPitStopTime (secunde, valori curate 17–60s)
  Modele:   Linear Regression (baseline)
            Random Forest Regressor
            Gradient Boosting Regressor
  Metrici:  MAE, RMSE, R²
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────────────────────
# 1. ÎNCĂRCARE DATE
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  PASUL 1: Încărcare date")
print("=" * 60)

df = pd.read_csv("data.csv")
print(f"  Dataset original: {df.shape[0]} rânduri x {df.shape[1]} coloane")


# ─────────────────────────────────────────────────────────────
# 2. CURĂȚARE DATE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 2: Curățare date")
print("=" * 60)

# 2.1 Convertim AvgPitStopTime la numeric
df['AvgPitStopTime'] = pd.to_numeric(df['AvgPitStopTime'], errors='coerce')

# 2.2 Eliminăm rândurile fără target
df = df[df['AvgPitStopTime'].notna()].copy()
print(f"  Dupa eliminare NaN target: {len(df)} rânduri")

# 2.3 Eliminăm outlieri: pit stopuri normale sunt 17–60s
#     Valorile > 60s reprezinta incidente, safety car, probleme tehnice
df = df[df['AvgPitStopTime'] < 60].copy()
df = df[df['AvgPitStopTime'] > 15].copy()
print(f"  Dupa filtrare outlieri (15–60s): {len(df)} rânduri")

# 2.4 Eliminăm coloanele care cauzează data leakage
#     (derivate direct din AvgPitStopTime — corelatie ~1.0)
leakage_cols = [
    'Driver Aggression Score',  # corr = -0.997 cu target
    'Lap Time Variation',       # corr =  1.000 cu target
    'Fast Lap Attempts',        # corr = -1.000 cu target
    'Pit_Time',                 # acelasi timp, alt format
]
df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

# 2.5 Eliminăm coloane irelevante pentru model
drop_cols = ['Race Name', 'Date', 'Time_of_race', 'Location',
             'Country', 'Abbreviation']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 3: Feature engineering")
print("=" * 60)

# 3.1 Rata pit stop-ului în cursă (laps relativ la total laps)
df['Pit_Lap_ratio'] = df['Pit_Lap'] / df['Laps']

# 3.2 Performanța istorică a constructorului (median pit stop per echipă)
constructor_median = df.groupby('Constructor')['AvgPitStopTime'].transform('median')
df['Constructor_pit_median'] = constructor_median

# 3.3 Clasificare compus — grupăm compușii vechi cu cei moderni
compound_map = {
    'HYPERSOFT': 'SOFT', 'ULTRASOFT': 'SOFT', 'SUPERSOFT': 'SOFT',
    'SOFT': 'SOFT', 'MEDIUM': 'MEDIUM', 'HARD': 'HARD',
    'INTERMEDIATE': 'WET', 'WET': 'WET', 'UNKNOWN': 'MEDIUM'
}
df['Tire_Group'] = df['Tire Compound'].map(compound_map).fillna('MEDIUM')
print("  Tire Groups create:", df['Tire_Group'].value_counts().to_dict())

# 3.4 Flag sezon modern (2022+ = noi regulamente aerodinamice)
df['Modern_Era'] = (df['Season'] >= 2022).astype(int)


# ─────────────────────────────────────────────────────────────
# 4. PREGĂTIRE FEATURES FINALE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 4: Pregătire features")
print("=" * 60)

FEATURES = [
    # Context cursă
    'Season', 'Round', 'Laps', 'Position', 'Position Changes',
    # Pit stop info
    'TotalPitStops', 'Stint', 'Stint Length', 'Pit_Lap', 'Pit_Lap_ratio',
    # Condiții meteo
    'Air_Temp_C', 'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH',
    # Features noi
    'Constructor_pit_median', 'Tire Usage Aggression', 'Modern_Era',
    # Categorice (vor fi encodate)
    'Constructor', 'Tire_Group',
]

TARGET = 'AvgPitStopTime'

# Păstrăm doar coloanele existente
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Features folosite: {len(FEATURES)}")
print(f"  {FEATURES}")

df_model = df[FEATURES + [TARGET]].copy()
print(f"\n  Rânduri disponibile: {len(df_model)}")
print(f"  Target — medie: {df_model[TARGET].mean():.2f}s, "
      f"std: {df_model[TARGET].std():.2f}s, "
      f"min: {df_model[TARGET].min():.2f}s, "
      f"max: {df_model[TARGET].max():.2f}s")

# 4.1 Encoding categorice
le_constructor = LabelEncoder()
le_tire       = LabelEncoder()

df_model['Constructor'] = le_constructor.fit_transform(
    df_model['Constructor'].astype(str))
df_model['Tire_Group']  = le_tire.fit_transform(
    df_model['Tire_Group'].astype(str))

# 4.2 Imputare valori lipsă cu mediana coloanei
imputer = SimpleImputer(strategy='median')
df_model[FEATURES] = imputer.fit_transform(df_model[FEATURES])

print(f"\n  Valori lipsă după imputare: {df_model.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────
# 5. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 5: Split train/test")
print("=" * 60)

X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train: {X_train.shape[0]} rânduri ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]} rânduri ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scalare pentru Linear Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────
# 6. ANTRENARE MODELE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 6: Antrenare modele")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(
                             n_estimators=200,
                             max_depth=12,
                             min_samples_leaf=4,
                             random_state=42,
                             n_jobs=-1
                         ),
    'Gradient Boosting': GradientBoostingRegressor(
                             n_estimators=200,
                             max_depth=5,
                             learning_rate=0.05,
                             subsample=0.8,
                             random_state=42
                         ),
}

results = {}

for name, model in models.items():
    print(f"\n  Antrenare {name}...")

    # Folosim date scalate pentru LR, originale pentru tree models
    if name == 'Linear Regression':
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
    }

    print(f"    MAE  = {mae:.3f} s")
    print(f"    RMSE = {rmse:.3f} s")
    print(f"    R²   = {r2:.4f}")


# ─────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION pe cel mai bun model
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 7: Cross-validation (k=5)")
print("=" * 60)

# Identificăm cel mai bun model după R²
best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"  Cel mai bun model: {best_name}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model, X, y, cv=kf,
    scoring='r2', n_jobs=-1
)

print(f"  R² per fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Medie R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

cv_mae = -cross_val_score(
    best_model, X, y, cv=kf,
    scoring='neg_mean_absolute_error', n_jobs=-1
)
print(f"  Medie MAE: {cv_mae.mean():.3f} ± {cv_mae.std():.3f} s")


# ─────────────────────────────────────────────────────────────
# 8. TABEL COMPARATIV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 8: Tabel comparativ final")
print("=" * 60)

print(f"\n  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("  " + "-" * 53)
for name, res in results.items():
    marker = " <-- cel mai bun" if name == best_name else ""
    print(f"  {name:<25} {res['MAE']:>7.3f}s {res['RMSE']:>7.3f}s {res['R2']:>8.4f}{marker}")


# ─────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 9: Feature importance")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(f"\n  Top 10 features ({best_name}):")
    for _, row in feat_df.head(10).iterrows():
        bar = '█' * int(row['Importance'] * 200)
        print(f"  {row['Feature']:<28} {row['Importance']:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# 10. VIZUALIZĂRI
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 10: Generare grafice...")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("F1 Pit Stop Time — Regression Analysis", fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = {'Linear Regression': '#4C72B0', 'Random Forest': '#DD8452', 'Gradient Boosting': '#55A868'}

# ── Grafic 1: Distribuția targetului ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(y, bins=50, color='#4C72B0', alpha=0.8, edgecolor='white', linewidth=0.4)
ax1.axvline(np.mean(y), color='red', linestyle='--', linewidth=1.5, label=f'Medie: {np.mean(y):.1f}s')
ax1.axvline(np.median(y), color='orange', linestyle='--', linewidth=1.5, label=f'Mediană: {np.median(y):.1f}s')
ax1.set_title("Distribuția AvgPitStopTime", fontsize=11)
ax1.set_xlabel("Timp (secunde)")
ax1.set_ylabel("Frecvență")
ax1.legend(fontsize=8)

# ── Grafic 2: Tabel metrici ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
table_data = [['Model', 'MAE', 'RMSE', 'R²']]
for name, res in results.items():
    short = name.replace('Regression', 'Reg.').replace('Gradient Boosting', 'Grad. Boost')
    table_data.append([short, f"{res['MAE']:.3f}s", f"{res['RMSE']:.3f}s", f"{res['R2']:.4f}"])

tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2.0)

# Evidențiem cel mai bun model
best_idx = list(results.keys()).index(best_name) + 1
for j in range(4):
    tbl[best_idx, j].set_facecolor('#d4edda')
    tbl[best_idx, j].set_text_props(fontweight='bold')
for j in range(4):
    tbl[0, j].set_facecolor('#343a40')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

ax2.set_title("Comparatie metrici modele", fontsize=11)

# ── Grafic 3: Bar chart R² ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
names_short = ['Linear Reg.', 'Random\nForest', 'Gradient\nBoosting']
r2_vals = [results[n]['R2'] for n in results]
bar_colors = [colors[n] for n in results]
bars = ax3.bar(names_short, r2_vals, color=bar_colors, alpha=0.85, edgecolor='white')
ax3.set_ylim(0, 1.05)
ax3.set_title("R² per model", fontsize=11)
ax3.set_ylabel("R²")
for bar, val in zip(bars, r2_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── Grafice 4–6: Predicted vs Actual pentru fiecare model ─────
for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
    y_pred = res['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, color=colors[name])
    lim_min = min(y_test.min(), y_pred.min()) - 1
    lim_max = max(y_test.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.2, label='Ideal')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    short = name.replace('Gradient Boosting', 'Grad. Boosting')
    ax.set_title(f"{short}\nMAE={res['MAE']:.3f}s  R²={res['R2']:.4f}", fontsize=10)
    ax.set_xlabel("Valori reale (s)")
    ax.set_ylabel("Valori prezise (s)")
    ax.legend(fontsize=8)

# ── Grafic 7: Reziduuri cel mai bun model ─────────────────────
ax7 = fig.add_subplot(gs[2, 0])
residuals = y_test - results[best_name]['y_pred']
ax7.scatter(results[best_name]['y_pred'], residuals, alpha=0.25, s=8, color=colors[best_name])
ax7.axhline(0, color='red', linestyle='--', linewidth=1.2)
ax7.set_title(f"Reziduuri — {best_name}", fontsize=11)
ax7.set_xlabel("Valori prezise (s)")
ax7.set_ylabel("Reziduuri (s)")

# ── Grafic 8: Feature importance ──────────────────────────────
ax8 = fig.add_subplot(gs[2, 1:])
if hasattr(best_model, 'feature_importances_'):
    top_feats = feat_df.head(12)
    ax8.barh(top_feats['Feature'][::-1], top_feats['Importance'][::-1],
             color=colors[best_name], alpha=0.85, edgecolor='white')
    ax8.set_title(f"Feature Importance — {best_name}", fontsize=11)
    ax8.set_xlabel("Importanță")
    for j, (val, feat) in enumerate(zip(top_feats['Importance'][::-1], top_feats['Feature'][::-1])):
        ax8.text(val + 0.001, j, f'{val:.4f}', va='center', fontsize=8)
else:
    ax8.axis('off')
    ax8.text(0.5, 0.5, 'Feature importance\nnedisponibil\npentru Linear Regression',
             ha='center', va='center', transform=ax8.transAxes, fontsize=11)

plt.savefig("f1_pitstop_results.png", dpi=150, bbox_inches='tight', facecolor='white')
print("  Grafice salvate: f1_pitstop_results.png")


# ─────────────────────────────────────────────────────────────
# 11. EXEMPLU PREDICȚIE PE DATE NOI
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PASUL 11: Exemplu predicție pe date noi")
print("=" * 60)

# Simulăm un pit stop: Mercedes, tur 30, anvelopă MEDIUM, vreme caldă
example = pd.DataFrame([{
    'Season': 2024, 'Round': 10, 'Laps': 57, 'Position': 3,
    'Position Changes': 1, 'TotalPitStops': 2, 'Stint': 1,
    'Stint Length': 28, 'Pit_Lap': 30, 'Pit_Lap_ratio': 30/57,
    'Air_Temp_C': 28.0, 'Track_Temp_C': 42.0, 'Humidity_%': 45.0,
    'Wind_Speed_KMH': 10.0, 'Constructor_pit_median': 23.71,
    'Tire Usage Aggression': 0.12, 'Modern_Era': 1,
    'Constructor': le_constructor.transform(['Mercedes'])[0],
    'Tire_Group': le_tire.transform(['MEDIUM'])[0],
}])

example_vals = example[FEATURES].values
example_imp  = imputer.transform(example_vals)

pred_lr  = models['Linear Regression'].predict(scaler.transform(example_imp))[0]
pred_rf  = models['Random Forest'].predict(example_imp)[0]
pred_gb  = models['Gradient Boosting'].predict(example_imp)[0]

print("\n  Scenariu: Mercedes, Tur 30/57, MEDIUM, 2024")
print(f"  Linear Regression  → {pred_lr:.2f}s")
print(f"  Random Forest      → {pred_rf:.2f}s")
print(f"  Gradient Boosting  → {pred_gb:.2f}s")
print(f"  Media reala Mercedes in dataset: 23.71s")

print("\n" + "=" * 60)
print("  GATA! Verifică f1_pitstop_results.png pentru grafice.")
print("=" * 60)