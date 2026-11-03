from pathlib import Path

import yfinance as yf

root=Path(__file__).resolve().parent

out=root/'SPY_2015_2025.csv'

if out.exists():
    raise SystemExit(f'{out.name} already exists. Back it up or remove it explicitly before redownloading.')

df=yf.download('SPY',start='2015-01-01',end='2026-01-01',interval='1d',auto_adjust=True,multi_level_index=False)

if df.empty:raise RuntimeError('No data returned')
df.to_csv(out)
print('Saved',len(df),'rows to',out)
