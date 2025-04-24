import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from fuzzywuzzy import process
import logging
import ast
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TMDB API configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Load the data
try:
    logger.info("Loading cosine similarity matrix...")
    cosine_sim = np.load('cosine_sim.npy')
    logger.info("Loading movie metadata...")
    movies_metadata_df = pd.read_csv('movies_metadata-1.csv')
    logger.info("Data loaded successfully")
    
    # Log available columns
    logger.info(f"Available columns: {movies_metadata_df.columns.tolist()}")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

# Clean and prepare the data
try:
    logger.info("Cleaning and preparing data...")
    movies_metadata_df['original_title'] = movies_metadata_df['original_title'].fillna('')
    
    # Handle optional columns
    optional_columns = {
        'overview': '',
        'vote_average': 0,
        'release_date': '',
        'genres': '[]',
        'tmdb_id': ''
    }
    
    for col, default in optional_columns.items():
        if col in movies_metadata_df.columns:
            movies_metadata_df[col] = movies_metadata_df[col].fillna(default)
        else:
            movies_metadata_df[col] = default
            logger.warning(f"Column '{col}' not found in dataset, using default value: {default}")
    
    logger.info("Data preparation completed")
except Exception as e:
    logger.error(f"Error preparing data: {str(e)}")
    raise

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class MovieTitle(BaseModel):
    title: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_movie_poster(title, year=None):
    """Fetch movie poster from TMDB API"""
    try:
        # Search for the movie
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'year': year
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        results = response.json()['results']
        if results:
            # Get the first result's poster path
            poster_path = results[0].get('poster_path')
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    except Exception as e:
        logger.error(f"Error fetching poster for {title}: {str(e)}")
    return None

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend/")
async def get_recommendations(movie: MovieTitle):
    try:
        logger.info(f"Getting recommendations for: {movie.title}")
        title = movie.title.lower()
        
        # Find the best match using fuzzy matching
        logger.info("Starting fuzzy matching...")
        best_match = process.extractOne(title, movies_metadata_df['original_title'].str.lower().tolist())
        logger.info(f"Best match found: {best_match}")
        
        if best_match[1] > 60:  # If we have a good match
            logger.info("Good match found, getting recommendations...")
            matched_movie = movies_metadata_df.loc[movies_metadata_df['original_title'].str.lower() == best_match[0]].iloc[0]
            idx = movies_metadata_df.loc[movies_metadata_df['original_title'].str.lower() == best_match[0]].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]  # Get top 5 similar movies
            
            recommendations = []
            for i in sim_scores:
                movie_data = movies_metadata_df.iloc[i[0]]
                try:
                    genres = ast.literal_eval(movie_data['genres']) if pd.notna(movie_data['genres']) else []
                except:
                    genres = []
                
                # Get release year for poster search
                release_year = None
                if movie_data['release_date']:
                    try:
                        release_year = int(movie_data['release_date'].split('-')[0])
                    except:
                        pass
                
                # Fetch poster URL
                poster_url = get_movie_poster(movie_data['original_title'], release_year)
                
                recommendations.append({
                    'title': movie_data['original_title'],
                    'similarity_score': float(i[1]),
                    'overview': movie_data['overview'],
                    'vote_average': float(movie_data['vote_average']),
                    'release_date': movie_data['release_date'],
                    'genres': genres,
                    'poster_url': poster_url
                })
            logger.info(f"Returning {len(recommendations)} recommendations")
            return {
                "matched_movie": matched_movie['original_title'],
                "recommendations": recommendations
            }
        else:
            logger.warning(f"No good match found for: {title}")
            return {"error": "Movie not found. Please try a different title."}
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

