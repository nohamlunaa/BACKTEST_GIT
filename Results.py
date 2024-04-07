import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from Indicators import Measures, Trend, Volatility, Adjustments
import seaborn as sns




class GraphAdjustments :

    # Fonction pour formatter l'axe Y en échelle log sans notation scientifique
    def log_format(val, pos=None):
        if val == 0:
            return "1"
        else:
            return f'{10**val:,.0f}'


'''
class Divers :

     def dessiner_courbes_equite_avec_colormap(equity_curves):
        # 1. Calculer les valeurs finales pour chaque courbe d'équité
        final_values = equity_curves.iloc[-1]
        # 2. Calculer les percentiles des valeurs finales
        equity_percentiles = final_values.rank(pct=True)
        # 3. Définir le colormap
        colors = ["red", "yellow", "green", "blue"]  # Définir plus de couleurs si souhaité
        cmap_name = 'my_colormap'
        colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        # 4. Dessiner les courbes d'équité avec coloration basée sur les percentiles
        plt.figure(figsize=(14, 7))
        for column in equity_curves.columns:
            # Trouver le percentile de la courbe actuelle pour déterminer sa couleur
            perc = equity_percentiles[column]
            color = colormap(perc)
            plt.plot(equity_curves.index, np.log10(equity_curves[column]), label=column, color=color)
        plt.title('Equity Curve for Each Indicator Type Across All Assets and Periods')
        plt.xlabel('Date')
        plt.ylabel('Log-Scale Equity')
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        plt.legend()
        plt.tight_layout()
        plt.show()

        



    # Extraction des noms simplifiés des indicateurs à partir des noms de colonnes
    # Le format est 'Equity_{NomIndicateur}_All_Periods'
    simplified_labels = [label.split('_')[1] for label in equity_curves.columns]
    # Calcul des rendements quotidiens pour chaque equity curve
    equity_returns = equity_curves.pct_change()
    # Calcul de la matrice de corrélation pour ces rendements
    correlation_matrix = equity_returns.corr()
    # Affichage de la heatmap avec les labels simplifiés
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05, xticklabels=simplified_labels, yticklabels=simplified_labels)
    plt.title('Matrice de Corrélation des Rendements des Equity Curves')
    # Calculer la moyenne des corrélations pour chaque indicateur
    correlation_means = correlation_matrix.mean()
    # Calculer les percentiles pour chaque moyenne de corrélation
    percentiles = correlation_means.rank(pct=True)
    # Extraire et simplifier les labels des indicateurs
    simplified_labels = [label.split('_')[1] for label in correlation_means.index]
    # Tri des indicateurs par leurs percentiles pour l'ordre des barres
    sorted_indices = percentiles.argsort()
    sorted_means = correlation_means.iloc[sorted_indices]
    sorted_labels = [simplified_labels[i] for i in sorted_indices]
    sorted_percentiles = percentiles.iloc[sorted_indices]
    # Définition des couleurs aux extrêmes et création de la LinearSegmentedColormap
    colors = ["blue", "green", "yellow", "red"]
    cmap_name = 'quantile_cmap'
    quantile_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)  # Utiliser 100 pour une transition plus douce



    # Création de la figure et des axes avec plt.subplots()
    fig, ax = plt.subplots(figsize=(14, 7))
    # Créer l'histogramme vertical sur l'axe principal avec les couleurs basées sur les percentiles
    for i, (mean, perc) in enumerate(zip(sorted_means, sorted_percentiles)):
        ax.bar(i, mean, color=quantile_cmap(perc), edgecolor='grey')
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=90)
    ax.set_xlabel('Indicateurs')
    ax.set_ylabel('Corrélation Moyenne')
    ax.set_title('Histogramme des Corrélations Moyennes avec Gradient de Couleurs Basé sur les Percentiles')
    # Création et ajout de la colorbar basée sur les percentiles
    sm = plt.cm.ScalarMappable(cmap=quantile_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Percentile de Corrélation')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    plt.tight_layout()


    # Étape 1: Calcul des Rolling Returns
    rolling_returns_columns = []  # Initialisation de la liste pour stocker les noms des nouvelles colonnes de rolling returns
    for column in equity_curves.columns:
        new_column_name = f'{column}_Rolling_Returns'
        equity_curves[new_column_name] = equity_curves[column].pct_change(periods=Adjustments.TRADING_DAYS_PER_YEAR) * Adjustments.PERCENTAGE_FACTOR  # Supposons 252 jours de trading par an et * 100 pour le pourcentage
        rolling_returns_columns.append(new_column_name)  # Ajoutez le nom de la nouvelle colonne à la liste
    # Étape 2: Calcul de la Moyenne des Rolling Returns
    # Utilisez la liste des colonnes de rolling returns spécifiquement créée pour éviter les confusions
    rolling_returns_mean = {column: equity_curves[column].mean() for column in rolling_returns_columns}
    # Étape 3: Calcul des Percentiles des Moyennes
    rolling_returns_percentiles = pd.Series(rolling_returns_mean).rank(pct=True)
    # Étape 4: Création de la Color Map
    colors = ["red", "yellow", "green", "blue"]
    cmap_name = 'rolling_returns_colormap'
    colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    # Étape 5: Graphique des Rolling Returns
    plt.figure(figsize=(14, 7))
    for column in rolling_returns_columns:  # Utilisez ici la liste des colonnes de rolling returns
        # Trouver le percentile pour la moyenne de rolling return pour déterminer sa couleur
        perc = rolling_returns_percentiles[column]
        color = colormap(perc)
        plt.plot(equity_curves.index, equity_curves[column], label=column.replace('Equity_', '').replace('_All_Periods_Rolling_Returns', ''), color=color)
    plt.title('Rolling Returns annuels pour chaque Equity Curve')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Rolling Returns (%)')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()


    plt.show()
'''