import polars as pl
import numpy as np

data = pl.read_csv("zz500_stocks.csv")

def calculate_factors(df):
    # 收盘价
    close = df['close'].to_numpy()
    
    # 市值
    df = df.with_columns(
        (pl.col('market_cap')).alias('MarketCap')
    )
    
    # 换手率
    df = df.with_columns(
        ((pl.col('volume') / pl.col('float_shares')).alias('TurnoverRate'))
    )
    
    # 市盈率
    df = df.with_columns(
        (pl.col('pe_ratio')).alias('PE')
    )
    
    # 动量因子
    df = df.with_columns(
        ((pl.col('close').pct_change(60)).alias('Momentum3M'))
    )
    
    # 成交量变化率
    df = df.with_columns(
        ((pl.col('volume').rolling_mean(5) / pl.col('volume').rolling_mean(20)).alias('VolumeChange'))
    )
    
    # 波动率因子
    df = df.with_columns(
        ((pl.col('close').rolling_std(60)).alias('Volatility'))
    )
    
    # 反转因子
    df = df.with_columns(
        ((pl.col('close').pct_change(20) / pl.col('close').pct_change(252)).alias('Reversal'))
    )
    
    return df

# 批量因子质量评估
def evaluate_factors(df):
    future_returns = df['future_returns'].to_numpy()

    ic_results = {}
    
    factor_columns = ['MarketCap', 'TurnoverRate', 'PE', 'Momentum3M', 'VolumeChange', 'Volatility', 'Reversal']
    
    for factor in factor_columns:
        factor_values = df[factor].to_numpy()
        ic = np.corrcoef(factor_values, future_returns)[0, 1]
        ic_results[factor] = ic
    
    return ic_results

df_factors = calculate_factors(data)

ic_results = evaluate_factors(df_factors)

print(ic_results)
