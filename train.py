import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('/Users/a123/Desktop/Baryon Count/physics_particles.csv')

cat_list = []
for label, content in df.items():
  if pd.api.types.is_string_dtype(content):
    cat_list.append(label)

num_list = []
for label, content in df.items():
  if pd.api.types.is_numeric_dtype(content):
    num_list.append(label)

df1 = df[cat_list].drop(['pdg_name','name','quarks'], axis=1)
df1 = df1.apply(pd.to_numeric, errors='coerce')
df1.astype(float)

df2 = df[cat_list].iloc[:,:3]


le = LabelEncoder()

ldf = df2.apply(le.fit_transform)

DF = pd.concat([ldf, df1, df[num_list]], axis=1)

for label, content in DF.items():
  if DF[label].isna().any():
    DF[label] = content.fillna(content.median())

DF.drop(['mass_lower', 'width_lower'], axis=1, inplace=True)

X = DF.drop('rank', axis=1)
Y = DF['rank']

np.random.seed(42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model0 = LogisticRegression()
model0.fit(X_train, Y_train)

joblib.dump(model0, 'Baryon_counter.joblib')
