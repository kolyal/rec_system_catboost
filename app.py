import os
from typing import List
from fastapi import FastAPI
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from catboost import Pool
from pydantic import BaseModel

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

app = FastAPI()

# ЗАГРУЗКА МОДЕЛИ
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model")
    from_file = CatBoostClassifier()

    loaded_model = from_file.load_model(model_path)

    return loaded_model

# ЗАГРУЗКА ФИЧЕЙ
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        """
        креды
        """
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
     return batch_load_sql('SELECT * FROM knaumov_user_features_lesson_22 LIMIT 1;')

engine = create_engine(
        """
        креды
        """
    )

val_data = load_features().drop('target', axis=1)
posts_df = pd.read_sql("""
SELECT * FROM post_text_df
""", con=engine)
cat_model = load_models()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    user_info = pd.read_sql("""
    SELECT * FROM user_data
    WHERE user_id = %s
    LIMIT 1;
    """, con=engine, params=(id,))

    posts_df['user_id'] = user_info['user_id']
    posts_df['user_id'].fillna(user_info['user_id'][0], inplace=True)
    posts_df['gender'] = user_info['gender']
    posts_df['gender'].fillna(user_info['gender'][0], inplace=True)
    posts_df['age'] = user_info['age']
    posts_df['age'].fillna(user_info['age'][0], inplace=True)
    posts_df['country'] = user_info['country']
    posts_df['country'].fillna(user_info['country'][0], inplace=True)
    posts_df['city'] = user_info['city']
    posts_df['city'].fillna(user_info['city'][0], inplace=True)
    posts_df['exp_group'] = user_info['exp_group']
    posts_df['exp_group'].fillna(user_info['exp_group'][0], inplace=True)
    posts_df['os'] = user_info['os']
    posts_df['os'].fillna(user_info['os'][0], inplace=True)
    posts_df['source'] = user_info['source']
    posts_df['source'].fillna(user_info['source'][0], inplace=True)

    new_order = val_data.columns.values

    cat_features = ['country', 'city', 'os', 'source', 'topic', 'text']

    pool = Pool(
        data=posts_df[new_order],
        cat_features=cat_features
    )

    preds = cat_model.predict_proba(pool)[:, 1]
    posts_df['score'] = preds
    posts_df_recs = posts_df.sort_values(by='score', ascending=False).head(limit)
    recommendations = [
        PostGet(id=int(rec['post_id']), text=str(rec['text']), topic=str(rec['topic']))
        for _, rec in posts_df_recs.iterrows()
    ]
    return recommendations
