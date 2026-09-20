"""
Screener Bot (versi GitHub Actions) — EMA 13/21 Golden Cross + SuperTrend
+ Volume Naik Minimal 1x
===========================================================================
Versi ini didesain untuk dijalankan sebagai SATU KALI CEK per eksekusi
(bukan loop terus-menerus), supaya cocok dipicu berkala oleh GitHub
Actions (cron job di cloud). Dengan begini, kamu TIDAK PERLU laptop,
PC, atau HP yang nyala terus — semuanya jalan di server GitHub secara
gratis, dan HP kamu cuma dipakai untuk:
    1. Setup awal (sekali saja, lewat browser)
    2. Menerima notifikasi Telegram

Bot ini SCAN SEMUA SAHAM IDX (bukan watchlist manual) — daftar saham
dibaca dari file idx_tickers.csv (hasil download resmi dari situs IDX,
BUKAN scraping otomatis — IDX melarang scraping di Terms of Use mereka,
jadi file ini harus kamu download & upload manual, lihat README).

Kriteria sinyal:
    1. EMA 13 golden cross EMA 21
    2. SuperTrend sedang bullish (hijau)
    3. Volume naik minimal 1x (2x lipat) dari volume bar sebelumnya
"""

import json
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
import requests

# ============================== CONFIG ==============================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

WATCHLIST_CSV = "idx_tickers.csv"   # daftar semua saham IDX, lihat README cara dapetinnya


def load_watchlist(path=WATCHLIST_CSV):
    """Baca daftar kode saham dari file CSV (hasil download resmi dari IDX).
    Fleksibel terhadap nama kolom — cari kolom yang mengandung kata 'kode',
    kalau tidak ketemu pakai kolom pertama."""
    if not os.path.exists(path):
        print(f"[ERROR] File {path} tidak ditemukan. Lihat README untuk cara mendapatkannya.")
        return []

    df = pd.read_csv(path)
    kode_col = None
    for col in df.columns:
        if "kode" in col.lower() or "code" in col.lower() or "ticker" in col.lower():
            kode_col = col
            break
    if kode_col is None:
        kode_col = df.columns[0]

    tickers = (
        df[kode_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    tickers = [f"{t}.JK" for t in tickers if t and t.isalpha()]
    print(f"[INFO] Watchlist dimuat: {len(tickers)} saham dari {path}")
    return tickers

TIMEFRAME = "15m"
EMA_FAST = 13
EMA_SLOW = 21
ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
VOLUME_MULTIPLIER = 2.0   # "naik minimal 1x" = volume sekarang >= 2x volume sebelumnya

STATE_FILE = "alert_state.json"

WIB = ZoneInfo("Asia/Jakarta")
MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(15, 50)

# ============================ END CONFIG =============================


def is_market_open():
    now_wib = datetime.now(WIB)
    if now_wib.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_wib.time() <= MARKET_CLOSE


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diisi (cek GitHub Secrets).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code != 200:
            print(f"[WARN] Gagal kirim Telegram: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Exception saat kirim Telegram: {e}")


def get_price_data(ticker, timeframe=TIMEFRAME, lookback="5d"):
    df = yf.download(ticker, period=lookback, interval=timeframe, progress=False)
    if df.empty:
        return None
    df = df.rename(columns=str.lower)
    return df


def get_batch_price_data(tickers, timeframe=TIMEFRAME, lookback="5d", batch_size=50):
    """Ambil data banyak saham sekaligus per batch, jauh lebih cepat
    dibanding download satu-satu (penting kalau watchlist isinya ratusan
    saham hasil scan semua IDX)."""
    results = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                tickers=batch, period=lookback, interval=timeframe,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as e:
            print(f"[ERROR] Gagal download batch {i}-{i+batch_size}: {e}")
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[ticker]
                df = df.dropna(how="all")
                if df.empty:
                    continue
                df = df.rename(columns=str.lower)
                results[ticker] = df
            except (KeyError, Exception):
                continue
    return results


def compute_ema(df, fast=EMA_FAST, slow=EMA_SLOW):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def compute_supertrend(df, period=ATR_PERIOD, multiplier=SUPERTREND_MULTIPLIER):
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
            continue

        prev_supertrend = supertrend.iloc[i - 1]

        if close.iloc[i] > prev_supertrend:
            direction.iloc[i] = 1
        elif close.iloc[i] < prev_supertrend:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(lowerband.iloc[i], prev_supertrend) if direction.iloc[i - 1] == 1 else lowerband.iloc[i]
        else:
            supertrend.iloc[i] = min(upperband.iloc[i], prev_supertrend) if direction.iloc[i - 1] == -1 else upperband.iloc[i]

    df["supertrend"] = supertrend
    df["st_direction"] = direction
    return df


def check_setup(ticker, df, state):
    if df is None or len(df) < max(EMA_SLOW, ATR_PERIOD) + 2:
        print(f"[INFO] {ticker}: data tidak cukup, skip.")
        return

    df = compute_ema(df)
    df = compute_supertrend(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    golden_cross_now = (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"])
    supertrend_bullish = last["st_direction"] == 1
    supertrend_just_flipped = prev["st_direction"] == -1 and last["st_direction"] == 1

    prev_volume = prev["volume"]
    last_volume = last["volume"]
    volume_ratio = (last_volume / prev_volume) if prev_volume > 0 else 0
    volume_confirmed = volume_ratio >= VOLUME_MULTIPLIER

    setup_match = golden_cross_now and supertrend_bullish and volume_confirmed

    bar_timestamp = str(df.index[-1])
    already_alerted = state.get(ticker) == bar_timestamp

    if setup_match and not already_alerted:
        price = last["close"]
        msg = (
            f"🟢 *SETUP MATCH: {ticker.replace('.JK', '')}*\n"
            f"Harga: {price:,.0f}\n"
            f"EMA13: {last['ema_fast']:,.1f} | EMA21: {last['ema_slow']:,.1f}\n"
            f"SuperTrend: {'Bullish (baru flip)' if supertrend_just_flipped else 'Bullish'}\n"
            f"Volume: naik {volume_ratio:.1f}x dari bar sebelumnya "
            f"({last_volume:,.0f} vs {prev_volume:,.0f})\n"
            f"Timeframe: {TIMEFRAME}\n"
            f"Waktu bar (WIB): {bar_timestamp}"
        )
        send_telegram_alert(msg)
        state[ticker] = bar_timestamp
        print(f"[ALERT] {ticker} -> notif terkirim")
    else:
        print(
            f"[INFO] {ticker}: belum match (golden_cross={golden_cross_now}, "
            f"supertrend_bullish={supertrend_bullish}, volume_ratio={volume_ratio:.2f}x)"
        )


def main():
    if not is_market_open():
        print(f"[INFO] Market tutup (sekarang {datetime.now(WIB)} WIB). Tidak melakukan cek.")
        return

    print(f"[INFO] Market buka ({datetime.now(WIB)} WIB). Mulai cek watchlist...")
    watchlist = load_watchlist()
    if not watchlist:
        print("[ERROR] Watchlist kosong, tidak ada yang dicek.")
        return

    print(f"[INFO] Mengambil data untuk {len(watchlist)} saham (per batch)...")
    price_data = get_batch_price_data(watchlist)
    print(f"[INFO] Data berhasil diambil untuk {len(price_data)} dari {len(watchlist)} saham.")

    state = load_state()
    for ticker, df in price_data.items():
        try:
            check_setup(ticker, df, state)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
    save_state(state)


if __name__ == "__main__":
    main()
