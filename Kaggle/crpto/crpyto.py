import tensorflow as tf
import pandas as pd 

df = pd.read_csv("Data/crypto_data/BCH-USD.csv", names=["time","low","high","open","close","volume"])

main_df = pd.DataFrame()
ratios = ["BTC-USD","LTC-USD","ETH-USD","BCH-USD"]
for ratio in ratios:
	dataset = f"Data/crypto_data/{ratio}.csv"
	df = pd.read_csv(dataset, names=["time","low","high","open","close","volume"])
