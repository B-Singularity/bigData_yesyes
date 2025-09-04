import pandas as pd

inputFile= 'preprocessing/population_long.csv'
df = pd.read_csv(inputFile)

print(df.head())