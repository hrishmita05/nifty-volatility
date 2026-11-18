# NIFTY50 Implied Volatility Surface

Builds a daily implied volatility surface for NIFTY50 options from NSE F&O bhavcopy data.

## Pipeline
1. `src/download_bhav.py`: downloads daily bhavcopy files into `bhav/`.
2. `src/nifty_iv_surface.py`: inverts Black-Scholes for implied vols, removes no-arbitrage violations, fits SVI and quadratic smiles per expiry, and writes results to `iv_output/`.

## Setup
    pip install -r requirements.txt

## Run
    python src/download_bhav.py
    python src/nifty_iv_surface.py --bhav_dir ./bhav --plot

## Notes
- Uses settlement prices, a fixed risk-free rate (6.5%) and dividend yield (1.3%).
- Data is not included; it is fetched from NSE.