# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:58:59 2021

@author: ramisedu
"""


import pandas as pd

df=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

print(df.to_string())


myds={"Cars":["Benz","Volvo"],
      "Color":["Black","White"]
      }

print(myds)

mydf=pd.DataFrame(myds)

print(mydf)

print(pd.__version__)

a=[1,7,2]

myvar=pd.Series(a,index=["x","y","z"])

print(myvar)

print(myvar["x"])

calories={"day1":350,"day2":370}

myvar=pd.Series(calories,index=["day1","day2"])

print(myvar["day1"])

calories={"day":[350,370,390],
           "duration":[50,60,70],
           "Age":[20,21,22]
         }

mydf=pd.DataFrame(calories)

print(mydf)


print(mydf.loc[0])

print(mydf.loc[0:2])


df=pd.DataFrame([pd.read_json("D:/ILearn/Datascience_Python/jsontest.json",typ='series')])

print(df)


data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)

print(df) 

print(df.head(2))

print(df.tail(2))

print(df.info())


mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

mydf.dropna(inplace=True)

print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

mydf.fillna("NA",inplace=True)

print(mydf.to_string())


mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")


mydf["Age"].fillna(45,inplace=True)

print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

x=mydf["Age"].mean()

mydf["Age"].fillna(x,inplace=True)

print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

x=mydf["Age"].median()

mydf["Age"].fillna(x,inplace=True)

print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

x=mydf["Age"].mode()[0]

mydf["Age"].fillna(x,inplace=True)

print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

print(mydf.to_string())

mydf.dropna(subset=["dateofbirth"],inplace=True)

mydf["dateofbirth"]=pd.to_datetime(mydf["dateofbirth"])

print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")





for x in mydf.index:
    if mydf.loc[x,"Age"]>45:
        mydf.loc[x,"Age"]=mydf["Age"].mean()
        
print(mydf.to_string())

mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")

for x in mydf.index:
    if mydf.loc[x,"Age"]>100:
        mydf.drop(x,inplace=True)
        
print(mydf.to_string())


mydf=pd.read_csv("D:/ILearn/Datascience_Python/csvtest.csv")
print(mydf.to_string())
print(mydf.duplicated())

mydf.drop_duplicates(inplace=True)

print(mydf)

print(mydf.corr())


data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df=pd.DataFrame(data)

print(df.to_string())

import matplotlib.pyplot as plt

df.plot()

plt.show()

df.plot(kind='scatter',x="Duration",y="Calories")
plt.show()

df.plot(kind='scatter',x="Duration",y="Maxpulse")
plt.show()


df["Duration"].plot(kind='hist')
plt.show()







