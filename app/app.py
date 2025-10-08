# app.py
import gradio as gr
import pickle
import pandas as pd
from pydantic import BaseModel, Field, conint, confloat
from typing import Dict

# 🎯 Modeli yükləyirik
with open("models\model.pkl", "rb") as f:
    model = pickle.load(f)

# 🎬 Bütün janrlar (modeldə istifadə olunan one-hot encoded feature-lər)
ALL_GENRES = [
    'Adventure', 'Comedy', 'Action', 'Mystery', 'Crime', 'Thriller',
    'Drama', 'Animation', 'Children', 'Horror', 'Documentary',
    'Sci-Fi', 'Fantasy', 'Film-Noir', 'Western', 'Musical', 'Romance',
    '(no genres listed)', 'War'
]

# 🎯 Pydantic input modeli
class MovieFeatures(BaseModel):
    movieYear: conint(ge=1900, le=2025) = Field(..., description="Filmin çıxış ili")
    userViews: conint(ge=1) = Field(..., description="İstifadəçinin baxdığı film sayı")
    userMeans: confloat(ge=0, le=5) = Field(..., description="İstifadəçinin orta reytinqi")
    genres: Dict[str, bool] = Field(..., description="Filmin janrları (seçilənlər 1 olacaq)")

# 🎯 Proqnoz funksiyası
def predict_movie_rating(movieYear, userViews, userMeans, genres):
    # Janr checkbox nəticələrini dict şəklində alırıq
    genre_dict = {genre: 1 if genre in genres else 0 for genre in ALL_GENRES}

    # Pydantic ilə validasiya
    features = MovieFeatures(
        movieYear=movieYear,
        userViews=userViews,
        userMeans=userMeans,
        genres=genre_dict
    )

    # DataFrame halına salırıq
    df = pd.DataFrame([{
        "movieYear": features.movieYear,
        "userViews": features.userViews,
        "userMeans": features.userMeans,
        **features.genres
    }])

    # Modeldən proqnoz alırıq
    pred = model.predict(df)[0]
    return f"Təxmini reytinq: {pred:.2f}"

# 🎨 Gradio interfeysi
genre_checkboxes = gr.CheckboxGroup(
    choices=ALL_GENRES,
    label="Filmin janr(lar)ını seç",
    info="Birdən çox janr seçmək olar"
)

demo = gr.Interface(
    fn=predict_movie_rating,
    inputs=[
        gr.Number(label="Filmin çıxış ili", value=2020),
        gr.Number(label="İstifadəçinin baxdığı film sayı", value=50),
        gr.Slider(label="İstifadəçinin orta reytinqi", minimum=0.0, maximum=5.0, value=3.5, step=0.1),
        genre_checkboxes
    ],
    outputs=gr.Textbox(label="Model nəticəsi"),
    title="🎥 Movie Rating Predictor (Content-Based)",
    description="Bu tətbiq CatBoost modelindən istifadə edərək, filmə veriləcək reytinqi təxmin edir.",
)

if __name__ == "__main__":
    demo.launch()
