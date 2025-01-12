import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the data
cosine_sim = np.load('cosine_sim.npy')
movies_metadata_df = pd.read_csv('movies_metadata-1.csv')

app = FastAPI()

class MovieTitle(BaseModel):
    title: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/recommend/")
async def read_movie_recommendations(movie: MovieTitle):
    recommendations = get_recommendations(movie.title, cosine_sim)
    return {"recommendations": recommendations}

def get_recommendations(title, cosine_sim):
    title = title.lower()
    if title in movies_metadata_df['original_title'].str.lower().values:
        idx = movies_metadata_df.loc[movies_metadata_df['original_title'].str.lower() == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Skip the first one
        movie_indices = [i[0] for i in sim_scores]
        return list(movies_metadata_df['original_title'].iloc[movie_indices])
    else:
        return ["Movie title not found in the dataset. Please check the title and try again."]

