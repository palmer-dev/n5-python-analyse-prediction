# =============================
# == ÉTAPE 0 ==
# Import des librairies
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns


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
plt.plot(df['Date'], df['Close'], label='Prix de clôture', color='blue')
plt.title('Évolution du cours de clôture du CAC 40')
plt.xlabel('Date')
plt.ylabel('Cours de clôture (€)')
plt.legend()
plt.grid()

# Graphique de l'évolution de la volatilité
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Volatility'], label='Volatilité', color='green')
plt.title('Évolution de la volatilité du CAC 40')
plt.xlabel('Date')
plt.ylabel('Volatilité  (%)')
plt.legend()
plt.grid()

# Correlation entre les colonnes Close, Volume et Volatility
corr = df[['Close', 'Volume', 'Volatility']].corr()
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Corrélation entre prix, volume et volatilité")

# Graphique de l'évolution du cours de clôture
plt.figure(figsize=(8, 4.5))
plt.plot(df['Date'], df['Return'], label='Rendement', color='blue')
plt.title('Évolution des rendements journaliers du CAC 40')
plt.xlabel('Date')
plt.ylabel('Cours de clôture (€)')
plt.legend()
plt.grid()

# Histogramme des rendements
plt.figure(figsize=(8, 4.5))
plt.hist(df['Return'].dropna(), bins=np.arange(-1, 1, 0.01), label="Rendement journalier", color='orange', edgecolor='black')
plt.title("Distribution des rendements journaliers du CAC 40")
plt.xlabel("Rendement (%)")
plt.ylabel("Fréquence")
plt.grid(True)

cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_hover(sel):
    x, y = sel.target
    sel.annotation.set_text(f"Date: {df['Date'].iloc[int(sel.index)]:%Y-%m-%d}\nValeur: {y:.8f}")

plt.show()