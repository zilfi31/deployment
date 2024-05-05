import pandas as pd

df = pd.read_csv("/Volumes/Data/deployment/klasifikasi/bank.csv")

buying_price = df['Buying_Price'].unique()
maintenance_price = df['Maintenance_Price'].unique()
nb_doors = df['No_of_Doors'].unique()
person_capacity = df['Person_Capacity'].unique()
size_of_luggage = df['Size_of_Luggage'].unique()
safety = df['Safety'].unique()