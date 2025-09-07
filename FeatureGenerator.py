import pandas as pd

class FeatureGenerator:
    def __init__(self, df: pd.DataFrame):
        """
        df: DataFrame with columns ['Open','High','Low','Close','Volume']
        """
        self.df = df.copy()

    def SMA(self, window=20):
        return self.df['Close'].rolling(window=window).mean().rename(f'SMA_{window}')

    def EMA(self, window=20):
        return self.df['Close'].ewm(span=window, adjust=False).mean().rename(f'EMA_{window}')

    def MACD(self, short=12, long=26, signal=9):
        short_ema = self.df['Close'].ewm(span=short, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=long, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({
            'MACD_Line': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Hist': hist
        })

    def RSI(self, window=14):
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # avoid div by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename(f'RSI_{window}')

    def ROC(self, window=12):
        """Rate of Change"""
        roc = self.df['Close'].pct_change(periods=window) * 100
        return roc.rename(f'ROC_{window}')

    def ATR(self, window=14):
        high, low, close = self.df['High'], self.df['Low'], self.df['Close']
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        return atr.rename(f'ATR_{window}')

    def BollingerBands(self, window=20, num_std=2):
        sma = self.df['Close'].rolling(window=window).mean()
        std = self.df['Close'].rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return pd.DataFrame({
            f'BB_Middle_{window}': sma,
            f'BB_Upper_{window}': upper,
            f'BB_Lower_{window}': lower
        })

    def StochasticOscillator(self, window=14, k_smooth=3, d_smooth=3):
        low_min = self.df['Low'].rolling(window=window).min()
        high_max = self.df['High'].rolling(window=window).max()
        rsv = (self.df['Close'] - low_min) / (high_max - low_min + 1e-10) * 100

        k = rsv.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        return pd.DataFrame({
            f'StochK_{window}': k,
            f'StochD_{window}': d
        })

    def KDJ(self, window=14, k_smooth=3, d_smooth=3, j_weight=3):
        stoch = self.StochasticOscillator(window, k_smooth, d_smooth)
        k, d = stoch.iloc[:, 0], stoch.iloc[:, 1]
        j = j_weight * k - 2 * d
        return pd.DataFrame({
            f'KDJ_K_{window}': k,
            f'KDJ_D_{window}': d,
            f'KDJ_J_{window}': j
        })

    def OBV(self):
        """On-Balance Volume"""
        obv = (self.df['Volume'] * 
              ( (self.df['Close'] > self.df['Close'].shift(1)) * 1
              - (self.df['Close'] < self.df['Close'].shift(1)) * 1 )).cumsum()
        return obv.rename('OBV')

    def CMF(self, window=20):
        """Chaikin Money Flow"""
        mfm = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'] + 1e-10)
        mfv = mfm * self.df['Volume']
        cmf = mfv.rolling(window=window).sum() / self.df['Volume'].rolling(window=window).sum()
        return cmf.rename(f'CMF_{window}')

    def all_indicators(self):
        features = pd.DataFrame(index=self.df.index)
        features[self.SMA(20).name] = self.SMA(20)
        features[self.SMA(50).name] = self.SMA(50)
        features[self.SMA(200).name] = self.SMA(200)
        features[self.EMA(20).name] = self.EMA(20)
        features[self.EMA(50).name] = self.EMA(50)
        features[self.EMA(200).name] = self.EMA(200)
        features = pd.concat([features, self.MACD()], axis=1)
        features[self.RSI(7).name] = self.RSI(7)
        features[self.RSI(14).name] = self.RSI(14)
        features[self.RSI(21).name] = self.RSI(21)
        features[self.ROC(12).name] = self.ROC(12)
        features[self.ATR(14).name] = self.ATR(14)
        features = pd.concat([features, self.BollingerBands()], axis=1)
        features = pd.concat([features, self.StochasticOscillator()], axis=1)
        features = pd.concat([features, self.KDJ()], axis=1)
        features[self.OBV().name] = self.OBV()
        features[self.CMF(20).name] = self.CMF(20)
        features[self.Change().name] = self.Change()
        return features
