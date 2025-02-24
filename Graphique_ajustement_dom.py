# Importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Charger les fichiers Excel
chemin_inverse = "/Users/elizabethfecteau/Documents/GPH-2006/Lab2/5/inverse.xlsx"
chemin_direct = "/Users/elizabethfecteau/Documents/GPH-2006/Lab2/5/direct.xlsx"
df_inverse = pd.read_excel(chemin_inverse, usecols=[1, 2], skiprows=1, names=["V", "I"], decimal=",")
df_direct = pd.read_excel(chemin_direct, usecols=[1, 2], skiprows=1, names=["V", "I"], decimal=",")

# Conversion explicite en numérique
df_inverse['V'] = pd.to_numeric(df_inverse['V'], errors='coerce')
df_inverse['I'] = pd.to_numeric(df_inverse['I'], errors='coerce')
df_direct['V'] = pd.to_numeric(df_direct['V'], errors='coerce')
df_direct['I'] = pd.to_numeric(df_direct['I'], errors='coerce')

# Nettoyage des données en supprimant les NaN
df_inverse = df_inverse.dropna()
df_direct = df_direct.dropna()

V_inverse = df_inverse["V"].values
I_inverse = df_inverse["I"].values
V_direct = df_direct["V"].values
I_direct = df_direct["I"].values

# Vérification des données chargées
print("Fichier inverse (5 premières lignes) :")
print(df_inverse.head())
print("\nFichier direct (5 premières lignes) :")
print(df_direct.head())

# Réduction du bruit avec une moyenne mobile
def moyenne_mobile(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

I_inverse_lisse = moyenne_mobile(I_inverse, window_size=5)
I_direct_lisse = moyenne_mobile(I_direct, window_size=5)

# Fusionner les données pour obtenir une courbe allant de -6V à 1V
V_totale = np.concatenate((-V_inverse[::-1], V_direct))  # On inverse V_inverse pour l'ordre croissant
I_totale = np.concatenate((-I_inverse_lisse[::-1], I_direct_lisse))  # Utilisation des données lissées

# Définition du modèle de Shockley
def equation_shockley(V, I0, V0):
    return I0 * (np.exp(V / V0) - 1)

# Ajustement des paramètres I0 et V0 avec les données lissées
params, covariance = curve_fit(equation_shockley, V_direct, I_direct_lisse, p0=[1e-12, 0.025])

I0_opt, V0_opt = params
print(f"Paramètres ajustés : I0 = {I0_opt}, V0 = {V0_opt}")

# Générer une courbe avec l'équation ajustée
V_fit = np.linspace(-6, 1, 500)  # Plage de tension complète
I_fit = equation_shockley(V_fit, I0_opt, V0_opt)

# Tracé du graphique
plt.figure(figsize=(8, 6))
plt.scatter(V_totale, I_totale, color='red', label="Données mesurées lissées", s=10)  # Points expérimentaux lissés
plt.plot(V_fit, I_fit, color='blue', label="Ajustement Shockley")  # Courbe ajustée
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel("Tension V (V)")
plt.ylabel("Courant I (A)")
plt.title("Caractéristique i-v de la diode avec ajustement Shockley (données lissées)")
plt.legend()
plt.grid()
plt.show()

