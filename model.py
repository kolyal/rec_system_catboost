import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

engine = create_engine(
    """
    креды
    """
)


df = pd.read_sql("""
SELECT * FROM feed_data f
LEFT JOIN user_data u
ON f.user_id = u.user_id
LEFT JOIN post_text_df p
ON f.post_id = p.post_id
LIMIT 200000;
"""
                 , con=engine)

df_sorted = df.sort_values(by='timestamp')
df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
df_sorted = df_sorted.drop('timestamp', axis=1)
df_sorted = df_sorted[df_sorted['action'] == 'view']
df_sorted = df_sorted.drop('action', axis=1)
X = df_sorted.drop('target', axis=1)
y = df_sorted['target']
X = X.loc[:, ~X.columns.duplicated()]
X_train, y_train = X.iloc[:round(0.75*X.shape[0])], y.iloc[:round(0.75*X.shape[0])]
X_test, y_test = X.iloc[round(0.75*X.shape[0]):], y.iloc[round(0.75*X.shape[0]):]
cat_cols = X.select_dtypes('object').columns.values

catboost_model = CatBoostClassifier(learning_rate=0.02, auto_class_weights='Balanced')
catboost_model.fit(X_train, y_train, cat_features=cat_cols)

catboost_model.save_model('catboost_model',
                           format="cbm")

from_file = CatBoostClassifier()

from_file.load_model("catboost_model")
print(precision_score(y_train, from_file.predict(X_train)))
print(precision_score(y_test, from_file.predict(X_test)))
print(f1_score(y_train, from_file.predict(X_train)))
print(f1_score(y_test, from_file.predict(X_test)))
