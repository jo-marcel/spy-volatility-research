from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "SPY_2015_2025.csv"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
WINDOW = 20
TRADING_DAYS = 252
TEST_START = "2023-01-01"
FIRST_WALK_YEAR = 2019
LAST_WALK_YEAR = 2025

FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
