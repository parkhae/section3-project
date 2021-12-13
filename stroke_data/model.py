import pandas as pd
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import sqlite3

df = pd.read_csv('stroke_data\healthcare-dataset-stroke-data.csv')

df = df.dropna()
df = df[df.smoking_status != 'Unknown'] #결측치 제거

X_data = df[['gender','age','hypertension','heart_disease','bmi','smoking_status']]
y_data = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=2, stratify=y_data)

pipe = make_pipeline(
    OrdinalEncoder(),
    XGBClassifier(n_estimators=1000,
                  random_state=2,
                  n_jobs=-1,
                  max_depth=20,
                  learning_rate=0.1,
                  eval_metric='mlogloss',
                  scale_pos_weight=0.054
                 )
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

con = sqlite3.connect("s3project.db")
X_data.to_sql("X_data", con, if_exists="append", index=False)
y_data.to_sql("y_data", con, if_exists="append", index=False)
con.commit()

import pickle
with open('model.pkl', 'wb') as pickle_file:
    pickle.dump(pipe, pickle_file)