# =============================
# == ÉTAPE 0 ==
# Import des librairies
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =============================
# == ÉTAPE 1 ==
# Lecutre du fichier CSV
# =============================

df = pd.read_csv(
    'cac40.csv', 
    delimiter=',', 
    na_values=['ERR', ''], 
    parse_dates=['Date'], 
    date_format='ISO8601'
)


# =============================
# == ÉTAPE 2 ==
# Inspection et nettoyage des données
# =============================

print("Aperçu des données :")
print(df.head())
print("\nInformations sur les données :")
print(df.info())

# Remplacement des valeurs incorrectes
regex_dict = {'€': '', 'ERR': np.nan}
df["Open"] = df['Open'].replace(regex_dict, regex=True)
df["High"] = df['High'].replace(regex_dict, regex=True)
df["Low"] = df['Low'].replace(regex_dict, regex=True)
df["Close"] = df['Close'].replace(regex_dict, regex=True)

# Suppression des doublons et des valeurs manquantes
print("Nombre de lignes avant nettoyage :", len(df))
df = df.drop_duplicates()
# Suppression des lignes avec des valeurs manquantes
df = df.dropna()
print("Nombre de lignes après nettoyage :", len(df))

# Tri des données par date
df = df.sort_values('Date').reset_index(drop=True)

# Vérification des jours manquants pour vérifier la cohérence temporelle
periode= [df['Date'].min(), df['Date'].max()]
minDate, maxDate = periode

print(f"Période des données : {minDate} à {maxDate}")

date_range = pd.date_range(minDate, maxDate, freq='D')
missing_days = date_range.difference(df['Date'])

if len(missing_days) != 0:
    print(f"{len(missing_days)} jour(s) manquant(s) dans les données.")


# =============================
# == ÉTAPE 3 ==
# Optimisation des types de données
# =============================

# Conversion des chaînes de caractères en type catégorie
df["Index"] = df["Index"].astype("category")

# Conversion des entiers en types plus économes en mémoire
df["Volume"] = pd.to_numeric(df["Volume"]).astype("int32")

# Conversion des flottants en types plus économes en mémoire
df["Open"] = pd.to_numeric(df["Open"]).astype("float32")
df["High"] = pd.to_numeric(df["High"]).astype("float32")
df["Low"] = pd.to_numeric(df["Low"]).astype("float32")
df["Close"] = pd.to_numeric(df["Close"]).astype("float32")

# Ajout de la volatilité pour le graphique
df['Volatility'] = df['Close'].pct_change().rolling(5).std().astype("float32") * 100 # Multiplié par 100 pour l'affichage en pourcentage
df['Return'] = df['Close'].pct_change().astype("float32") * 100 # Multiplié par 100 pour l'affichage en pourcentage
 
print(df.info())


# =============================
# == ÉTAPE 4 ==
# Analyse visuelle de base via des graphiques
# =============================

# Graphique de l'évolution du cours de clôture
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Close'], label='Prix de clôture', color='red')
plt.title('Évolution du cours de clôture')
plt.xlabel('Date')
plt.ylabel('Cours de clôture (€)')
plt.legend()
plt.grid()
evolution_close = plt.gca() # Sauvegarde de l'axe pour l'utiliser dans la détection d'anomalies

# Graphique de l'évolution de la volatilité
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Volatility'], label='Volatilité', color='green')
plt.title('Évolution de la volatilité')
plt.xlabel('Date')
plt.ylabel('Volatilité  (%)')
plt.legend()
plt.grid()

# Graphique de l'évolution du rendement
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Return'], label='Rendement', color='blue')
plt.title('Évolution des rendements')
plt.xlabel('Date')
plt.ylabel('Rendement (%)')
plt.legend()
plt.grid()

# Correlation entre les colonnes Close, Volume et Volatility
corr = df[['Close', 'Volume', 'Volatility']].corr()
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Corrélation entre prix, volume et volatilité")

# Histogramme des rendements
plt.figure(figsize=(8, 4.5))
plt.hist(df['Return'].dropna(), bins=np.arange(-1, 1, 0.01), label="Rendement journalier", color='orange', edgecolor='black')
plt.title("Distribution des rendements")
plt.xlabel("Rendement (%)")
plt.ylabel("Fréquence")
plt.grid(True)

cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_hover(sel):
    x, y = sel.target
    sel.annotation.set_text(f"Date: {df['Date'].iloc[int(sel.index)]:%Y-%m-%d}\nValeur: {y:.8f}")


# =============================
# == ÉTAPE 5 ==
# Détection des anomalies basées sur les rendements et la volatilité
# =============================

z_thresh = 3.0  # seuil z-score sur les rendements
vol_mul = 10.0  # multiplicateur pour seuil de volatilité (mean + vol_mul*std)

# Z-score sur les rendements
ret_mean = df['Return'].mean()
ret_std  = df['Return'].std(ddof=0)
df['z_return'] = (df['Return'] - ret_mean) / (ret_std if ret_std != 0 else 1) # évite la division par zéro

# Seuil sur la volatilité
vol_mean = df['Volatility'].mean()
vol_std  = df['Volatility'].std(ddof=0)
vol_threshold = vol_mean + vol_mul * vol_std

# Condition d'anomalie : grand rendement OU pic de volatilité
df['anomaly_return'] = df['z_return'].abs() >= z_thresh
df['anomaly_vol'] = df['Volatility'] >= vol_threshold
df['anomaly'] = df['anomaly_return'] | df['anomaly_vol']

# Clean up des colonnes temporaires
df = df.drop(columns=['z_return', 'anomaly_return', 'anomaly_vol'])

print("Anomalies détectées:", df['anomaly'].sum())

# Extrait les points du DataFrame uniquement où il y a des anomalies
anoms = df[df['anomaly']]

# Superpose les anomalies sur le graphique de l'évolution du cours de clôture
evolution_close.scatter(anoms['Date'], anoms['Close'], color='red', s=40, zorder=5, label='Anomalies')


# =============================
# == ÉTAPE 6.1 ==
# Analyse statistique du CAC40
# =============================

# Calcule de la moyenne mobile sur 10 jours du cours de clôture
df['MA_10'] = df['Close'].rolling(10).mean()
# Calcule de la moyenne mobile sur 20 jours du cours de clôture
df['MA_20'] = df['Close'].rolling(20).mean()

# Calcule du Momentum long sur 20 jours
df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

# Enregistrer les données nettoyées et enrichies dans un nouveau fichier PARQUET
df.to_parquet('cac40_cleaned.parquet', index=False)


# =============================
# == ÉTAPE 6.2 ==
# Prédiction simple du cours
# =============================
main_prediction = 'prediction'
best_accuracy = 0.0
max_shift_day = 10

# Calcul de la précision de la prédiction
for shif_day in range(1, max_shift_day):
    df['prediction'] = ((df['MA_10'] > df['MA_20']) & (df['Close'] > df['Close'].shift(shif_day))).astype(int)
    df['actual'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['correct'] = (df['prediction'] == df['actual']).astype(int)
    accuracy = df['correct'].mean() * 100
    if ((accuracy > best_accuracy) & (shif_day > 1)):
        best_accuracy = accuracy
        main_prediction = f'prediction_{shif_day}'
    print(f"Précision de la prédiction à {shif_day} jour(s) : {accuracy:.2f}%")

print(f"La meilleure prédiction est la prédiction à {main_prediction} jour(s) avec une précision de {best_accuracy:.2f}%")

# Affichage de la prédiction la plus précise sur le graphique du cours de clôture
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Close'], label='Prix de clôture', color='grey')
plt.title('Évolution du cours de clôture')
plt.xlabel('Date')
plt.ylabel('Cours de clôture (€)')
plt.legend()
plt.grid()

# Flèches vers le haut pour prédiction = 1
plt.scatter(
    df['Date'][df['prediction']==1], 
    df['Close'][df['prediction']==1],
    marker='^', 
    color='green', 
    s=30, 
    zorder=5,
    label='Prédiction hausse'
)

# Flèches vers le bas pour prédiction = 0
plt.scatter(
    df['Date'][df['prediction']==0], 
    df['Close'][df['prediction']==0],
    marker='v', 
    color='red', 
    s=30, 
    zorder=5,
    label='Prédiction baisse'
)


# =============================
# == ÉTAPE 7 ==
# Prédiction avec Machine Learning (Random Forest Classifier)
# =============================

# Ajout de la moyenne mobile simple sur 5 jours
df['MA_5'] = df['Close'].rolling(5).mean()

# Ajout de l'Exponential Moving Average (EMA) sur 5 jours
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

# Ajout du Rate of Change (ROC) sur 5 jours
df['ROC_5'] = df['Close'].pct_change(5)

# Ajout du Momentum sur 5 jours
df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

# Ajout de la Volatility sur 5 jours
df['Volatility_5'] = df['Close'].pct_change().rolling(5).std()

# Ajout de la plage High-Low
df['High_Low_Range'] = df['High'] - df['Low']

# Ajout de la moyenne mobile simple du volume sur 5 jours
df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()

# Ajout de la moyenne mobile simple du volume sur 5 jours
df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()

# Définition des features et de la cible
feature_cols = ['MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'EMA_5', 
                'ROC_5', 'Volatility_5', 'High_Low_Range', 
                'Volume_SMA_5', 'Volume_SMA_10']

X = df[feature_cols]
y = (df['Close'].shift(-1) > df['Close']).astype(int)

# Division en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
accuracy = model.score(X_test, y_test)
print(f"Précision sur le jeu de test : {accuracy*100:.2f}%")

# Prédictions
df['ML_Prediction'] = model.predict(X)

# Crée un nouveau graphique pour afficher les prédictions ML
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Close'], label='Prix de clôture', color='grey')
plt.title('Évolution du cours de clôture')
plt.xlabel('Date')
plt.ylabel('Cours de clôture (€)')
plt.legend()
plt.grid()

# Affichage des prédictions ML sur le graphique du cours de clôture
plt.scatter(
    df['Date'][df['ML_Prediction']==1], 
    df['Close'][df['ML_Prediction']==1],
    marker='^', 
    color='lime',
    s=50, 
    zorder=6,
    label='ML Hausse'
)
plt.scatter(
    df['Date'][df['ML_Prediction']==0], 
    df['Close'][df['ML_Prediction']==0],
    marker='v', 
    color='red', 
    s=50, 
    zorder=6,
    label='ML Baisse'
)


# Affiche tous les graphiques
plt.show()