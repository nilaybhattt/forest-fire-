import streamlit as st
import pandas as pd
import pydeck as pdk
import json
import csv
import matplotlib.pyplot as plt
import networkx as net
from urllib.request import urlopen
from six.moves import urllib
from datetime import datetime

# with open ("Forest_Fire.csv","r") as f:
#         reader=csv.reader(f)
#         next(reader)
#         data={"Forest":[]}
#         for row in reader:
#             data["Forest"].append({"Area":row[0],"Oxygen":row[1],"Temperature":row[2],"Humidity":row[3],"Fire Occurence":row[4],"lat":row[5],"lon":row[6]})
# with open ("Forest.json","w") as f:
#     json.dump(data,f,indent=1)



st.set_option('deprecation.showPyplotGlobalUse', False)


df = pd.read_csv('pollution.csv')
df.columns
df= df.rename(columns = {" pm25": "pm25", 
                         " pm10":"pm10", 
                         " o3": "o3",
                         ' no2' : 'no2',
                         ' so2' : 'so2',
                         ' co' : 'co'})

df.columns
df['date'] = pd.to_datetime(df.date)
df21 = df.loc[df['date'] > '2019-10-31']
df21 = df21.sort_values(by = 'date')
df21.drop(13, inplace=True)
df21.replace(' ', '0', inplace=True)

dates = df21['date']
pm25 = df21['pm25']
pm25 = [int(i) for i in pm25]
o3 = df21['o3']
o3 = [int(i) for i in o3]
no2 = df21['no2']
no2 = [int(i) for i in no2]
so2 = df21['so2']
so2 = [int(i) for i in so2]


plt.figure(figsize=(10,8))
length = len(dates)
ploti = st.selectbox("", ["pm25","O3","NO2","SO2"])
if ploti=="pm25" :
    plt.plot(dates,pm25)
elif ploti=="O3":
    plt.plot(dates,o3)
elif ploti=="NO2":
    plt.plot(dates,no2)
if ploti=="SO2":
    plt.plot(dates,so2)

st.pyplot()

# mask = (df['date'] >= '2019-10-31') & (df['date']  < '2020-04-1')
# past21 = df.loc[mask]


# dates = df21['date']
# pm25_l = df21['pm25']
# pm25_l = [int(i) for i in pm25]


# pm25_n = past21['pm25']
# pm25_n = [int(i) for i in pm25_n]

# plt.figure(figsize=(10,8))

# length = [i for i in range(1,len(dates)+1)]

# plt.plot(length,pm25_l,color='blue',label='under lockdown')
# plt.plot(length,pm25_n,color='red',label='before lockdown')
# plt.legend()
# plt.title('Comparision of before lockdown vs under lockdown pm2.5 values')
# st.pyplot()