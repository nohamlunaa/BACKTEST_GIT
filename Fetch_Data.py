import pandas as pd
import numpy as np
import time
from Indicators import Adjustments

class CSV :


    def tradingview(file_path, close_asset_name, n_assets=None, specific_assets=None, n_days=None):

        # Charger et créer les données
        data = pd.read_csv(file_path, sep=None, engine='python', parse_dates=['time'], index_col='time')
        # Convertir l'index en UTC
        data.index = data.index.tz_localize('UTC')
        # Renommer l'index de 'time' en 'date'
        data.index.rename('date', inplace=True)

        # Renommer la colonne "close" en le nom de l'actif principal
        data.rename(columns={'close': close_asset_name}, inplace=True)

        # Initialiser asset_names vide
        asset_names = []

        # Si n_assets est spécifié comme 0, retourner uniquement l'index
        if n_assets == 0:
            return data.index, asset_names

        # Si specific_assets est spécifié, cette liste dictera les actifs à inclure
        if specific_assets:
            # Itérer à travers chaque specific_asset pour l'ajouter s'il est trouvé
            for specific_asset in specific_assets:
                found = False
                for column in data.columns:
                    # Extraction du nom de l'actif à partir de la colonne
                    asset_name = column.split('·')[0].strip() if '·' in column else column
                    if specific_asset == asset_name:
                        data[specific_asset] = data[column]
                        asset_names.append(specific_asset)
                        found = True
                        break
                if not found:
                    print(f"Warning: {specific_asset} not found in data.")
        else:
            # Si aucun specific_assets n'est fourni, comportement par défaut basé sur n_assets
            for column in data.columns:
                if column.endswith(('open', 'high', 'low')):
                    continue
                asset_name = column.split('·')[0].strip() if '·' in column else column
                if asset_name not in asset_names:
                    data[asset_name] = data[column]
                    asset_names.append(asset_name)
                if n_assets is not None and len(asset_names) >= n_assets:
                    break

        # Filtrer le DataFrame pour ne conserver que les colonnes des actifs sélectionnés
        data = data[asset_names]

        # Si n_days est spécifié, sélectionner les n_days dernières lignes
        if n_days is not None:
            data = data.tail(n_days)

        print(f"Actifs sélectionnés : {asset_names}")

        return data,asset_names


        






class RandomData :

    def simulate_asset_prices(n_days=1252, n_assets=2, sigma=0.20, drift_range=(-0.05, 0.05)):
        # Initialisation du générateur aléatoire
        np.random.seed(int(time.time()))
        total_days = int(n_days * (365.25 / Adjustments.TRADING_DAYS_PER_YEAR))
        today = pd.Timestamp.today()
        start_date = today - pd.Timedelta(days=total_days)
        # Paramètres de la simulation¨
        initial_price=100
        initial_prices = [initial_price] * n_assets  # Prix initial pour chaque actif
        mus = np.random.uniform(drift_range[0], drift_range[1], n_assets)  # Drifts aléatoires pour chaque actif
        dt = 1/252  # Pas de temps pour des données quotidiennes

        # Initialisation des prix
        prices = np.zeros((n_days, n_assets))
        prices[0] = initial_prices

        # Simulation des prix jour par jour
        for t in range(1, n_days):
            random_returns = np.random.normal((mus - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), (1, n_assets))
            prices[t] = prices[t-1] * np.exp(random_returns)

        # Générer automatiquement les noms pour les actifs
        asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

        # Créer un DataFrame avec les dates comme index
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        data = pd.DataFrame(prices, index=dates, columns=asset_names)

        # Retourner le DataFrame et les noms des colonnes
        return data, asset_names