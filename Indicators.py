import pandas as pd
import numpy as np
from scipy.stats import mode

class Adjustments :

    TRADING_DAYS_PER_WEEK = 5
    TRADING_DAYS_PER_MONT = 21
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_5_YEARS = TRADING_DAYS_PER_YEAR * 5
    ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR)
    PERCENTAGE_FACTOR = 100

class Measures : 
    
    

    def SMA(df, column_name, length):
        return df[column_name].rolling(window=length, min_periods=1).mean()
        

    def EMA(df, column_name, length):
        return df[column_name].ewm(span=length, adjust=False).mean()
    

    def WMA(df, column_name, length):
        weights = np.arange(1, length + 1)
        return df[column_name].rolling(window=length, min_periods=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    

    def median(df, column_name, length):
        return df[column_name].rolling(window=length, min_periods=1).median()
    

    def mode_average(df, column_name, length, rounding_precision):
        # Arrondir les valeurs à la précision spécifiée
        rounded_prices = df[column_name].round(rounding_precision)
        # Fonction pour calculer le mode et gérer les égalités
        def custom_mode(x):
            # Calculer le mode pour la fenêtre
            modes_res = mode(x, axis=None)
            # Extraire les modes
            modes = np.squeeze(modes_res.mode)
            # Si plusieurs modes, calculer et retourner la moyenne des modes
            # Vérifier si modes est un scalaire (ce qui signifie un seul mode)
            if np.isscalar(modes):
                return modes
            else:
                # Si ce n'est pas un scalaire, alors nous avons plusieurs modes et nous retournons leur moyenne
                return np.mean(modes)
        # Calculer le mode roulant en utilisant une fenêtre glissante
        rolling_mode = rounded_prices.rolling(window=length, min_periods=1).apply(custom_mode, raw=False)        
        return rolling_mode
    
    
    def ROC(df, column_name, length):
        roc = (df[column_name].diff(length) / df[column_name].shift(length))
        return roc


    def donchian_channels(df, column_name, length):
        upper = df[column_name].rolling(window=length).max()  
        lower = df[column_name].rolling(window=length).min() 
        basis = (upper + lower) / 2
        return basis
    

    def aroon(df, column_name, length):
        # Identifier l'index du prix de clôture le plus haut sur la période
        rolling_high_idx = df[column_name].rolling(window=length, min_periods=0).apply(lambda x: x.argmax(), raw=True)
        # Calculer Aroon Up comme le pourcentage de la période depuis le dernier prix de clôture le plus haut
        aroon_up = 100 * (length - rolling_high_idx) / length  
        # Identifier l'index du prix de clôture le plus bas sur la période
        rolling_low_idx = df[column_name].rolling(window=length, min_periods=0).apply(lambda x: x.argmin(), raw=True)
        # Calculer Aroon Down comme le pourcentage de la période depuis le dernier prix de clôture le plus bas
        aroon_down = 100 * (length - rolling_low_idx) / length
        return aroon_up, aroon_down


class Trend :

    def emaRatio(df, column_name, lenST, lenLT):
        emaST = Measures.EMA(df, column_name, lenST)
        emaLT = Measures.EMA(df, column_name, lenLT)
        ema_ratio = (emaST / emaLT) - 1
        return ema_ratio


    def smaRatio(df, column_name, lenST, lenLT):
        smaST = Measures.SMA(df, column_name, lenST)
        smaLT = Measures.SMA(df, column_name, lenLT)
        sma_ratio = (smaST / smaLT) - 1
        return sma_ratio


    def wmaRatio(df, column_name, lenST, lenLT):
        wmaST = Measures.WMA(df, column_name, lenST)
        wmaLT = Measures.WMA(df, column_name, lenLT)
        wma_ratio = (wmaST / wmaLT) - 1
        return wma_ratio
    

    def median_ratio(df, column_name, lenST, lenLT):        
        median_ST = Measures.median(df, column_name, lenST)
        median_LT = Measures.median(df, column_name, lenLT)
        median_ratio = (median_ST / median_LT) - 1
        return median_ratio
    

    def mode_ratio(df, column_name, lenST, lenLT, rounding_precision = 1):
        modeST = Measures.mode_average(df, column_name, lenST, rounding_precision)
        modeLT = Measures.mode_average(df, column_name, lenLT, rounding_precision)
        mode_ratio = (modeST / modeLT) - 1
        return mode_ratio 


    def ROC_WMA(df, column_name, lenST, lenLT):
        ma = Measures.WMA(df, column_name, lenST)
        roc = (ma - ma.shift(lenLT)) / ma.shift(lenLT)
        return roc


    def donchian_channel_ratio(df, column_name, lenST, lenLT):
        basisST = Measures.donchian_channels(df, column_name, lenST)
        basisLT = Measures.donchian_channels(df, column_name, lenLT)
        donchian_ratio = (basisST / basisLT) - 1
        return donchian_ratio


    def aroon_ratio(df, column_name, lenLT):
        aroon_up, aroon_down = Measures.aroon(df, column_name, lenLT)
        # Renvoie 1 si aroon_up est supérieur à aroon_down, sinon -1
        aroon_ratio = np.where(aroon_up < aroon_down, 1, -1)
        return aroon_ratio


    def cti(df, column_name, length):
        # Assurer que la période est au minimum 2
        length = max(2, length)
        def calc_cti(x):
            # Générer Y basé sur la position (inversement linéaire)
            Y = np.arange(-len(x) + 1, 1)
            # Calculs des composants nécessaires à la formule de corrélation
            Ex = x.sum()
            Ey = Y.sum()
            Ex2 = (x**2).sum()
            Ey2 = (Y**2).sum()
            Exy = (x * Y).sum()
            # Calcul du dénominateur
            denominator = (len(x) * Ex2 - Ex**2) * (len(x) * Ey2 - Ey**2)
            # Vérifier si le dénominateur est zéro
            if denominator == 0:
                return 0
            else:
                return (len(x) * Exy - Ex * Ey) / np.sqrt(denominator)
        # Application de la fonction sur une fenêtre glissante
        cti = df[column_name].rolling(window=length, min_periods=length).apply(calc_cti, raw=True)
        return cti
        


    def rsi_wma(df, column_name, lenST, lenLT):
        # Calculer la WMA de la colonne spécifiée
        wma = Measures.WMA(df, column_name, lenST)
        # Calculer le changement de la WMA
        wma_diff = wma.diff(1)
        # Séparer les gains et les pertes
        gains = wma_diff.where(wma_diff > 0, 0.0)
        losses = -wma_diff.where(wma_diff < 0, 0.0)
        # Calculer la moyenne des gains et des pertes
        avg_gain = gains.rolling(window=lenLT, min_periods=1).mean()
        avg_loss = losses.rolling(window=lenLT, min_periods=1).mean()
        # Éviter la division par zéro
        avg_loss = avg_loss.replace(0, 0.001)
        # Calculer le RS et le RSI, ajuster le RSI pour osciller entre -50 et 50
        rs = avg_gain / avg_loss
        rsi = 50 - (100 / (1 + rs))
        return rsi

    def adx_wma(df, column_name, lenST, lenLT):
        # Calculer la WMA sur la colonne spécifiée pour simuler les "high" et "low"
        wma = Measures.WMA(df, column_name, lenST)
        # Utiliser la WMA pour calculer les "high_col" et "low_col" simulés
        high_col = wma.rolling(window=lenLT).max()
        low_col = wma.rolling(window=lenLT).min()
        # Correction: Définir 'up' et 'down' avant leur utilisation
        up = high_col.diff()
        down = -low_col.diff()
        # Maintenant, utilisez 'up' et 'down' pour calculer plusDI et minusDI
        plusDI = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        minusDI = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        # Normalisation des DI+ et DI- par l'ATR
        plusDI_normalized = pd.Series(plusDI).rolling(window=lenLT).sum() 
        minusDI_normalized = pd.Series(minusDI).rolling(window=lenLT).sum() 
        # Calcul d'un "ratio ADX" simplifié en remplacement du calcul standard de l'ADX
        adx_ratio = (plusDI_normalized - minusDI_normalized) / (plusDI_normalized + minusDI_normalized)
        return adx_ratio
    
    def psych_index(df, column_name, lenST, lenLT) :
        threshold = 50
        wma = Measures.WMA(df, column_name, lenST)
        UpDay = (wma > wma.shift(1)).astype(int)
        PsychIndex = UpDay.rolling(window=lenLT).sum() / lenLT * 100
        Psy = PsychIndex - threshold
        return Psy


class Volatility :

    def calculate_hv(df, column_name, length):
        prices = df[column_name].replace(0, 0.001).ffill().clip(lower=1e-8)
        log_returns = np.log(prices / prices.shift(1))
        hv = log_returns.rolling(window=length).std() * Adjustments.ANNUALIZATION_FACTOR * Adjustments.PERCENTAGE_FACTOR
        return hv


    def calculate_composite_hv(df, column_name, lengths, multipliers):
        composite_hv = np.zeros(len(df))
        for length, multiplier in zip(lengths, multipliers):
            hv = Volatility.calculate_hv(df, column_name, length) * multiplier
            composite_hv += hv
        total_multiplier = np.nansum(multipliers)
        return composite_hv / total_multiplier