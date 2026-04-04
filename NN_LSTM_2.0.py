"""
ST7 — LSTM + RevIN Electricity Forecasting
==========================================
Parallel implementation to st7_forecaster.py (N-BEATS version).
Same data pipeline, same outputs, different model core:
  - Bidirectional LSTM (multi-layer) for temporal encoding
  - Temporal Self-Attention to re-weight history
  - Cross-Attention to fuse future exogenous weather
  - RevIN for distribution-shift robustness
  - MC Dropout for calibrated 90% confidence intervals
  - Early Stopping for overfitting prevention
  - Adam + L2 Weight Decay for regularization

Usage:
  python3 st7_forecaster_lstm.py --demo               # quick synthetic test
  python3 st7_forecaster_lstm.py --epochs 20          # full RTE data
  python3 st7_forecaster_lstm.py --rte_path <csv> --weather_path <csv> --epochs 20
"""

import os
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ==========================================
# 1. Reversible Instance Normalization (RevIN)
# ==========================================
class RevIN(nn.Module):
    """
    Instance-level normalization that saves μ/σ per window and denormalizes output.
    Kim et al. (ICLR 2022) — robust against seasonal distribution shift.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean  = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        return x * self.stdev + self.mean


# ==========================================
# 2. Temporal Self-Attention
# ==========================================
class TemporalAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention over the time axis.
    Allows the LSTM output to re-weight the importance of different hours
    in the 168h context window — key for handling weekday/weekend rhythms.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.query  = nn.Linear(hidden_dim, hidden_dim)
        self.key    = nn.Linear(hidden_dim, hidden_dim)
        self.value  = nn.Linear(hidden_dim, hidden_dim)
        self.scale  = hidden_dim ** -0.5
        self.out    = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, V)                               # (B, T, H)
        return self.out(out)


# ==========================================
# 3. LSTM + RevIN + Attention Model
# ==========================================
class LSTMWithRevIN(nn.Module):
    """
    Architecture:
      1. RevIN normalizes the full input history (168h × num_features)
      2. Bidirectional LSTM encodes temporal dynamics → hidden states (T, 2*hidden)
      3. Temporal Self-Attention re-weights hidden states by importance
      4. MLP decoder takes [last attention state | future exogenous (24h × (F-1))]
         and projects to the 24h forecast
      5. RevIN denormalizes output back to MW scale
    """
    def __init__(self, num_features, context_length, pred_len,
                 hidden_size=128, num_layers=2, target_idx=0, dropout_rate=0.3):
        super().__init__()
        self.target_idx     = target_idx
        self.pred_len       = pred_len
        self.hidden_size    = hidden_size
        self.revin_layer    = RevIN(num_features, affine=True)

        # Bidirectional LSTM — captures both forward and backward temporal patterns
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        lstm_out_dim = hidden_size * 2  # bidir doubles the output

        # Self-attention over LSTM outputs
        self.attention = TemporalAttention(lstm_out_dim, dropout_rate)
        self.attn_dropout = nn.Dropout(dropout_rate)

        # Future exogenous projection — compresses weather forecast to a vector
        future_input_dim = pred_len * (num_features - 1)
        self.future_proj = nn.Sequential(
            nn.Linear(future_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Final decoder: LSTM context + future exogenous → 24h forecast
        decoder_input = lstm_out_dim + hidden_size // 2
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, pred_len),
        )

    def forward(self, x, x_future):
        batch_size = x.size(0)

        # 1. RevIN normalise history
        x_norm = self.revin_layer(x, 'norm')

        # Save target stats for denormalization
        target_mean  = self.revin_layer.mean[:, :, self.target_idx:self.target_idx + 1].squeeze(-1)  # (B, 1)
        target_stdev = self.revin_layer.stdev[:, :, self.target_idx:self.target_idx + 1].squeeze(-1)
        if self.revin_layer.affine:
            t_weight = self.revin_layer.affine_weight[self.target_idx]
            t_bias   = self.revin_layer.affine_bias[self.target_idx]

        # 2. LSTM encoding: (B, T, num_features) → (B, T, 2*hidden)
        lstm_out, _ = self.lstm(x_norm)

        # 3. Temporal self-attention
        attn_out = self.attention(lstm_out)        # (B, T, 2*hidden)
        attn_out = self.attn_dropout(attn_out)
        context  = attn_out[:, -1, :]             # take last timestep as summary (B, 2*hidden)

        # 4. Future exogenous encoding
        x_fut_flat  = x_future.view(batch_size, -1)     # (B, pred_len*(F-1))
        future_vec  = self.future_proj(x_fut_flat)       # (B, hidden//2)

        # 5. Decode: context + future → forecast
        combined  = torch.cat([context, future_vec], dim=-1)   # (B, 2*hidden + hidden//2)
        pred_norm = self.decoder(combined)                      # (B, 24) in normalized scale

        # 6. Denormalize back to MW
        if self.revin_layer.affine:
            pred_norm = (pred_norm - t_bias) / (t_weight + self.revin_layer.eps ** 2)
        pred = pred_norm * target_stdev + target_mean

        return pred  # (B, 24)


# ==========================================
# 4. Dataset (identical to N-BEATS version)
# ==========================================
class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset.
    x       : (context_length, num_features)  — past history
    x_future: (pred_len, num_features-1)      — future weather only (no consumption)
    y       : (pred_len,)                     — future consumption to predict
    """
    def __init__(self, data, context_length, target_idx, pred_len=24):
        self.data           = torch.FloatTensor(np.array(data, copy=True))
        self.context_length = context_length
        self.target_idx     = target_idx
        self.pred_len       = pred_len

    def __len__(self):
        return len(self.data) - self.context_length - self.pred_len + 1

    def __getitem__(self, idx):
        x        = self.data[idx : idx + self.context_length, :]
        feat_idx = [i for i in range(self.data.shape[1]) if i != self.target_idx]
        x_future = self.data[idx + self.context_length : idx + self.context_length + self.pred_len, feat_idx]
        y        = self.data[idx + self.context_length : idx + self.context_length + self.pred_len, self.target_idx]
        return x, x_future, y


# ==========================================
# 5. Data Loading (identical to N-BEATS version)
# ==========================================
column_translation_weather = {
    "ID OMM station": "WMO_Station_ID", "Date": "Date",
    "Pression au niveau mer": "Sea_Level_Pressure_hPa",
    "Direction du vent moyen 10 mn": "Wind_Direction_10min_deg",
    "Vitesse du vent moyen 10 mn": "Wind_Speed_10min_mps",
    "Température": "Temperature_K", "Température (°C)": "Temperature_C",
    "Point de rosée": "Dew_Point_C", "Humidité": "Relative_Humidity_percent",
    "Nebulosité totale": "Total_Cloud_Cover_okta",
    "Latitude": "Latitude", "Longitude": "Longitude", "Altitude": "Altitude_m",
    "mois_de_l_annee": "Month_of_Year",
}

def load_and_clean_weather(file_path):
    df = pd.read_csv(file_path, sep=';') if file_path.endswith('.csv') else pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df.rename(columns=column_translation_weather, inplace=True)
    df.replace("ND", np.nan, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", utc=True).dt.tz_localize(None)
    df["Temperature_C"]  = pd.to_numeric(df.get("Temperature_C", df.get("Température (°C)")), errors="coerce")
    df["Altitude_m"]     = pd.to_numeric(df.get("Altitude_m", df.get("Altitude")), errors="coerce")
    cols_to_keep = ["Datetime", "WMO_Station_ID", "Temperature_C", "Altitude_m", "Latitude", "Longitude"]
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df.dropna(subset=["Datetime", "Temperature_C"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    for col, fn in [("Hour", lambda d: d.dt.hour), ("DayOfWeek", lambda d: d.dt.dayofweek),
                    ("Month", lambda d: d.dt.month), ("DayOfYear", lambda d: d.dt.dayofyear),
                    ("Year", lambda d: d.dt.year)]:
        df[col] = fn(df["Datetime"])
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    return df


def load_and_preprocess_data(rte_path=None, weather_path=None, demo=False):
    if demo or rte_path is None or weather_path is None:
        print("Using synthetic demo data...")
        t = np.arange(2000)
        consumption = 50000 + 10000 * np.sin(2 * np.pi * t / 24) + 5000 * np.sin(2 * np.pi * t / (24 * 365)) + np.random.normal(0, 1000, len(t))
        temperature = 15 + 10 * np.sin(2 * np.pi * (t - 6) / 24) + np.random.normal(0, 2, len(t))
        solar       = np.clip(800 * np.sin(2 * np.pi * (t - 6) / 24), 0, None) + np.random.normal(0, 50, len(t))
        df = pd.DataFrame({'consumption': consumption, 'temperature': temperature, 'solar_irradiance': solar})
        return df, 0

    try:
        rte_df = pd.read_csv(rte_path, skiprows=1)
        rte_df.columns = rte_df.columns.str.strip()
        rte_df['Datetime'] = pd.to_datetime(rte_df['Datetime'], errors='coerce')
        rte_df.dropna(subset=['Datetime'], inplace=True)
        rte_df.set_index('Datetime', inplace=True)
        rte_df = rte_df.select_dtypes(include=[np.number])
        print(f"Electricity: {rte_df.shape[0]} rows.")
    except Exception as e:
        print(f"Error loading electricity: {e}"); return None, None

    try:
        wx_df = pd.read_csv(weather_path, skiprows=1)
        wx_df.columns = wx_df.columns.str.strip()
        wx_df['Datetime'] = pd.to_datetime(wx_df['Datetime'], errors='coerce')
        wx_df.dropna(subset=['Datetime', 'Latitude', 'Temperature_C'], inplace=True)
        wx_df = wx_df[wx_df['Latitude'] > 40.0]

        def get_region(lat):
            if lat > 48.5: return 'R1_North'
            if lat > 46.5: return 'R2_CenterNorth'
            if lat > 44.5: return 'R3_CenterSouth'
            return 'R4_South'

        wx_df['Region'] = wx_df['Latitude'].apply(get_region)
        cols_to_pivot = ['Temperature_C', 'Hour', 'DayOfWeek', 'Month']
        wx_regional = wx_df.groupby(['Datetime', 'Region'])[cols_to_pivot].mean().reset_index()
        wx_pivot = wx_regional.pivot(index='Datetime', columns='Region', values='Temperature_C')
        wx_pivot.columns = [f'Temp_{c}' for c in wx_pivot.columns]
        weights = {'Temp_R1_North': 0.45, 'Temp_R2_CenterNorth': 0.20,
                   'Temp_R3_CenterSouth': 0.20, 'Temp_R4_South': 0.15}
        wx_pivot['Weighted_National_Temp'] = sum(wx_pivot.get(c, 0) * w for c, w in weights.items())
        wx_pivot.drop(columns=[c for c in wx_pivot.columns if c.startswith('Temp_R')], inplace=True, errors='ignore')
        wx_national = wx_df.groupby('Datetime')[['Altitude_m', 'Latitude', 'Longitude']].mean()
        weather_final = pd.merge(wx_pivot, wx_national, left_index=True, right_index=True)
        print(f"Weather: {weather_final.shape[0]} hourly rows, Regionalized {list(wx_pivot.columns)}")
    except Exception as e:
        print(f"Error regionalizing weather: {e}"); return None, None

    rte_hourly     = rte_df.resample('1h').mean()
    weather_hourly = weather_final.resample('1h').mean()
    merged_df = pd.merge(rte_hourly, weather_hourly, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.interpolate(method='linear', limit_direction='both').ffill().bfill().dropna()
    print(f"Merged: {merged_df.shape[0]} rows, {merged_df.shape[1]} features.")

    target_col = next((c for c in merged_df.columns if 'consumption' in c.lower() or 'consommation' in c.lower()), merged_df.columns[0])
    cols = [target_col] + [c for c in merged_df.columns if c != target_col]
    merged_df = merged_df[cols]
    for c in list(cols[1:]):
        mean, std = merged_df[c].mean(), merged_df[c].std()
        if std > 1e-5:
            merged_df[c] = (merged_df[c] - mean) / std

    return merged_df, 0


# ==========================================
# 6. Training
# ==========================================
class EarlyStopping:
    """Stop training when validation loss stops improving; restore best weights."""
    best_model_state: Optional[dict]

    def __init__(self, patience=20, min_delta=0):
        self.patience         = patience
        self.min_delta        = min_delta
        self.counter          = 0
        self.best_loss:  Optional[float] = None
        self.early_stop       = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss        = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter          = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, train_loader, val_loader, test_loader=None,
                num_epochs=20, lr=1e-3, device='cpu', patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.to(device)
    history       = {'train_loss': [], 'val_loss': [], 'test_loss': []}
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for batch_x, batch_x_fut, batch_y in train_loader:
            batch_x, batch_x_fut, batch_y = batch_x.to(device), batch_x_fut.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x, batch_x_fut)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # gradient clipping
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_x_fut, batch_y in val_loader:
                batch_x, batch_x_fut, batch_y = batch_x.to(device), batch_x_fut.to(device), batch_y.to(device)
                pred      = model(batch_x, batch_x_fut)
                val_loss += criterion(pred, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        # --- Test (passive tracking) ---
        test_loss = 0.0
        _tl: Optional[DataLoader] = test_loader
        if _tl is not None:
            with torch.no_grad():
                for batch_x, batch_x_fut, batch_y in _tl:
                    batch_x, batch_x_fut, batch_y = batch_x.to(device), batch_x_fut.to(device), batch_y.to(device)
                    pred       = model(batch_x, batch_x_fut)
                    test_loss += criterion(pred, batch_y).item() * batch_x.size(0)
            test_loss /= len(_tl.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if test_loader is not None:
            history['test_loss'].append(test_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_str = f" - Test Loss: {test_loss:.4f}" if test_loader else ""
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}{test_str}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights...")
            break

    if early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)

    return history


# ==========================================
# 7. MC Dropout Inference
# ==========================================
def predict_with_uncertainty(model, dataloader, num_samples=50, device='cpu'):
    """Run T stochastic forward passes with Dropout active to estimate uncertainty."""
    model.train()   # keeps dropout ON
    model.to(device)
    all_preds, actuals = [], []

    with torch.no_grad():
        for batch_x, batch_x_fut, batch_y in dataloader:
            batch_x, batch_x_fut = batch_x.to(device), batch_x_fut.to(device)
            batch_preds = np.stack(
                [model(batch_x, batch_x_fut).cpu().numpy() for _ in range(num_samples)],
                axis=1
            )  # (B, T, 24)
            all_preds.append(batch_preds)
            actuals.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)   # (N, T, 24)
    actuals   = np.concatenate(actuals,   axis=0)

    mean_preds  = np.mean(all_preds, axis=1)
    lower_bound = np.percentile(all_preds, 5,  axis=1)
    upper_bound = np.percentile(all_preds, 95, axis=1)
    return mean_preds, lower_bound, upper_bound, actuals


# ==========================================
# 8. Plotting & Reporting (same as N-BEATS)
# ==========================================
def plot_learning_curves(history, output_path='lstm_overfitting.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', lw=2)
    plt.plot(history['val_loss'],   label='Validation Loss', lw=2)
    if history.get('test_loss'):
        plt.plot(history['test_loss'], label='Test Loss', lw=2, linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('LSTM — Overfitting Analysis: Learning Curves')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()
    print(f"[Output] Overfitting analysis saved → {output_path}")


def generate_results_matrix(actuals, mean_preds, lower_ci, upper_ci, output_prefix='lstm_predictions'):
    act_flat   = actuals.flatten()
    pred_flat  = mean_preds.flatten()
    lower_flat = lower_ci.flatten()
    upper_flat = upper_ci.flatten()

    abs_err    = np.abs(act_flat - pred_flat)
    pct_err    = abs_err / np.where(act_flat == 0, 1, np.abs(act_flat)) * 100
    within_ci  = ((act_flat >= lower_flat) & (act_flat <= upper_flat)).astype(int)
    ci_width   = upper_flat - lower_flat

    results_df = pd.DataFrame({
        'Actual_MW':         act_flat, 'Predicted_Mean_MW': pred_flat,
        'Lower_CI_5pct_MW':  lower_flat, 'Upper_CI_95pct_MW': upper_flat,
        'Abs_Error_MW':      abs_err,    'Pct_Error':          pct_err,
        'Within_90pct_CI':   within_ci,  'CI_Width_MW':        ci_width,
    })
    csv_path = f'{output_prefix}_with_ci.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"[Output] Full predictions saved → {csv_path}")

    mae      = float(np.mean(abs_err))
    mape     = float(np.mean(pct_err))
    rmse     = float(np.sqrt(np.mean((act_flat - pred_flat) ** 2)))
    coverage = float(np.mean(within_ci) * 100)
    mean_ciw = float(np.mean(ci_width))
    bias     = float(np.mean(pred_flat - act_flat))

    metrics = {
        'MAE (MW)':            f'{mae:,.1f}',
        'RMSE (MW)':           f'{rmse:,.1f}',
        'MAPE (%)':            f'{mape:.2f}',
        'Mean Bias (MW)':      f'{bias:+,.1f}',
        '90% CI Coverage (%)': f'{coverage:.1f}',
        'Mean CI Width (MW)':  f'{mean_ciw:,.1f}',
        'Test Samples':        f'{len(actuals):,}',
    }
    summary_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    summary_csv = f'{output_prefix}_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Output] Summary metrics saved → {summary_csv}")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    # --- 4-panel figure ---
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f1117')
    COLORS = {'actual': '#e8eaed', 'pred': '#a78bfa', 'ci': '#a78bfa',
              'error': '#ef5350', 'text': '#e8eaed', 'grid': '#2a2d3e',
              'table_header': '#5b21b6', 'table_row_a': '#1a1d2e', 'table_row_b': '#12151f'}

    N_CHUNKS = max(1, min(15, len(actuals) // 24))
    act_plot  = actuals[:N_CHUNKS * 24:24].flatten()
    pred_plot = mean_preds[:N_CHUNKS * 24:24].flatten()
    low_plot  = lower_ci[:N_CHUNKS * 24:24].flatten()
    up_plot   = upper_ci[:N_CHUNKS * 24:24].flatten()
    xs = np.arange(len(act_plot))

    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.set_facecolor(COLORS['grid'])
    ax1.fill_between(xs, low_plot, up_plot, color=COLORS['ci'], alpha=0.18, label='90% CI')
    ax1.plot(xs, act_plot,  color=COLORS['actual'], lw=1.2, label='Actual')
    ax1.plot(xs, pred_plot, color=COLORS['pred'],   lw=1.2, label='Predicted', linestyle='--')
    ax1.set_title(f'Forecast vs Actual ({N_CHUNKS} days)', color=COLORS['text'], fontsize=12, pad=10)
    ax1.set_xlabel('Test Hour Index', color=COLORS['text']); ax1.set_ylabel('MW', color=COLORS['text'])
    ax1.tick_params(colors=COLORS['text'])
    ax1.legend(facecolor='#1a1d2e', labelcolor=COLORS['text'], fontsize=9)

    ax2 = fig.add_subplot(3, 2, 3)
    ax2.set_facecolor(COLORS['grid'])
    ax2.hist(abs_err, bins=60, color=COLORS['error'], alpha=0.8, edgecolor='none')
    ax2.axvline(mae, color='white', lw=1.5, linestyle='--', label=f'MAE = {mae:,.0f} MW')
    ax2.set_title('Absolute Error Distribution', color=COLORS['text'], fontsize=11)
    ax2.set_xlabel('|Error| (MW)', color=COLORS['text']); ax2.set_ylabel('Count', color=COLORS['text'])
    ax2.tick_params(colors=COLORS['text'])
    ax2.legend(facecolor='#1a1d2e', labelcolor=COLORS['text'], fontsize=9)

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.set_facecolor(COLORS['grid'])
    idx = np.random.choice(len(act_flat), min(1000, len(act_flat)), replace=False)
    ax3.scatter(act_flat[idx], pred_flat[idx], alpha=0.15, s=5, color=COLORS['pred'])
    lims = [min(act_flat.min(), pred_flat.min()), max(act_flat.max(), pred_flat.max())]
    ax3.plot(lims, lims, color='white', lw=1, linestyle='--', label='Perfect')
    ax3.set_title('Predicted vs Actual (scatter)', color=COLORS['text'], fontsize=11)
    ax3.set_xlabel('Actual (MW)', color=COLORS['text']); ax3.set_ylabel('Predicted (MW)', color=COLORS['text'])
    ax3.tick_params(colors=COLORS['text'])
    ax3.legend(facecolor='#1a1d2e', labelcolor=COLORS['text'], fontsize=9)

    ax4 = fig.add_subplot(3, 2, (5, 6)); ax4.axis('off'); ax4.set_facecolor('#0f1117')
    tbl = ax4.table(cellText=[[k, v] for k, v in metrics.items()],
                    colLabels=['Metric', 'Value'], cellLoc='left', loc='center',
                    bbox=[0.05, 0.0, 0.9, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#2a2d3e')
        if row == 0:  cell.set_facecolor(COLORS['table_header']); cell.set_text_props(color='white', fontweight='bold')
        elif row % 2: cell.set_facecolor(COLORS['table_row_b']); cell.set_text_props(color=COLORS['text'])
        else:         cell.set_facecolor(COLORS['table_row_a']); cell.set_text_props(color=COLORS['text'])

    fig.suptitle('ST7 — LSTM + RevIN Forecasting: Results Matrix',
                 color=COLORS['text'], fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    matrix_path = f'{output_prefix}_matrix.png'
    plt.savefig(matrix_path, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"[Output] Results matrix figure saved → {matrix_path}")

    rows_html = ''.join(f'<tr><td>{k}</td><td><strong>{v}</strong></td></tr>' for k, v in metrics.items())
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>ST7 LSTM Forecasting – Results</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e8eaed;margin:2rem 4rem;}}
  h1{{color:#a78bfa;}} h2{{color:#c4b5fd;border-bottom:1px solid #2a2d3e;padding-bottom:6px;}}
  table{{border-collapse:collapse;width:50%;margin-top:1rem;}}
  th{{background:#5b21b6;color:white;padding:10px 16px;text-align:left;}}
  td{{padding:8px 16px;border-bottom:1px solid #2a2d3e;}}
  tr:nth-child(even){{background:#1a1d2e;}}
  img{{max-width:100%;border-radius:8px;margin-top:1rem;}}
  .note{{color:#aaa;font-size:0.9rem;margin-top:0.5rem;}}
</style></head><body>
<h1>ST7 — LSTM + RevIN Electricity Forecasting</h1>
<p class="note">Generated on test set ({len(actuals):,} samples)</p>
<h2>Performance Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{rows_html}</table>
<h2>Results Matrix</h2>
<img src="{matrix_path}" alt="Results matrix">
<p class="note">90% CI Coverage should be close to 90%.</p>
</body></html>"""
    html_path = f'{output_prefix}_report.html'
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"[Output] HTML report saved → {html_path}")
    return results_df, summary_df


# ==========================================
# 9. Main Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST7 LSTM Electricity Forecaster")
    parser.add_argument('--demo',         action='store_true', help='Run on synthetic data')
    parser.add_argument('--rte_path',     type=str, default=None)
    parser.add_argument('--weather_path', type=str, default=None)
    parser.add_argument('--epochs',       type=int, default=20)
    args = parser.parse_args()

    # Configuration (Stress Test for Overfitting)
    CONTEXT_LENGTH = 72
    PRED_LEN       = 24
    HIDDEN_SIZE    = 128    # Increase capacity
    NUM_LAYERS     = 1
    DROPOUT_RATE   = 0.05   # Reduce regularization to allow overfitting
    BATCH_SIZE     = 128
    DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

    _script_dir   = os.path.dirname(os.path.abspath(__file__))
    _rte_path     = args.rte_path     or os.path.join(_script_dir, "electricity", "France_Electricity_Cleaned.csv")
    _weather_path = args.weather_path or os.path.join(_script_dir, "weather", "France_Weather_Cleaned.csv")

    df, target_idx = load_and_preprocess_data(rte_path=_rte_path, weather_path=_weather_path, demo=args.demo)
    if df is None:
        print("Failed to load data."); exit(1)
    assert df is not None  # reassure type checker: exit(1) above guarantees df is not None here

    data_array = df.values
    n          = len(data_array)
    train_data = data_array[:int(n * 0.70)]
    val_data   = data_array[int(n * 0.70):int(n * 0.85)]
    test_data  = data_array[int(n * 0.85):]
    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    train_ds = TimeSeriesDataset(train_data, CONTEXT_LENGTH, target_idx, PRED_LEN)
    val_ds   = TimeSeriesDataset(val_data,   CONTEXT_LENGTH, target_idx, PRED_LEN)
    test_ds  = TimeSeriesDataset(test_data,  CONTEXT_LENGTH, target_idx, PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    num_features = data_array.shape[1]
    model = LSTMWithRevIN(
        num_features=num_features,
        context_length=CONTEXT_LENGTH,
        pred_len=PRED_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        target_idx=target_idx,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Starting Training...")
    history = train_model(model, train_loader, val_loader, test_loader=test_loader,
                          num_epochs=args.epochs, device=DEVICE)

    output_prefix = 'demo_lstm_predictions' if args.demo else 'lstm_predictions'
    plot_learning_curves(history, output_path=f"{output_prefix}_overfitting.png")

    print("Evaluating on Test Set with MC Dropout...")
    mean_preds, lower_ci, upper_ci, actuals = predict_with_uncertainty(
        model, test_loader, num_samples=50, device=DEVICE
    )

    generate_results_matrix(actuals, mean_preds, lower_ci, upper_ci, output_prefix=output_prefix)