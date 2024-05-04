from flask import Flask, render_template, request, url_for, redirect, jsonify, session, flash
import requests
from urllib.parse import unquote
import pandas as pd
import ast
from ast import literal_eval
import random
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pickle
import openai
import csv

# Initialize Firebase Admin SDK
cred_path = 'moviereccomendation-1e6e3-firebase-adminsdk-ktnu3-44b66432fb.json'  # Update the path to your Firebase credentials
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
app.secret_key = 'a35fsw56uhg7ufjuh'  # Change this to your actual secret key


# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'sk-proj-xz1H9m4cUyBr0dKxdRHfT3BlbkFJKmD8yAHwNYun5ImeUrmm'

def generate_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-3-small"  # Using the specified model
    )
    # Extract the embeddings from the response data
    return [item['embedding'] for item in response['data']]





import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
df=pd.read_csv('combined_movies.csv')
title_overview = df[['title', 'movie_info']]

with open('movie_embeddings.pkl', 'rb') as f:
    loaded_embeddings = pickle.load(f)

# Assuming loaded_embeddings and movie_popularity are defined
titles, embeddings = zip(*loaded_embeddings)
embeddings_array = np.array(embeddings)
movie_popularity = df['popularity']# Example dictionary

# Clustering with K-means
num_clusters = 10  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings_array)

def generate_query_embedding(query_text):
    embedding = generate_embeddings([query_text])[0]  # Assuming this returns a list
    return np.array(embedding)  # Convert list to numpy array

def find_similar_movies(query_text):
    query_embedding = generate_query_embedding(query_text)
    query_cluster = kmeans.predict(query_embedding.reshape(1, -1))[0]
    cluster_indices = np.where(clusters == query_cluster)[0]

    # Using Nearest Neighbors within the cluster
    cluster_embeddings = embeddings_array[cluster_indices]
    local_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    local_model.fit(cluster_embeddings)
    distances, local_indices = local_model.kneighbors([query_embedding])

    # Retrieve titles and sort by popularity
    similar_movies = [titles[cluster_indices[local_index]] for local_index in local_indices[0]]
    sorted_movies = sorted(similar_movies, key=lambda x: movie_popularity.get(x, 0), reverse=True)

    # Print or return the similar movies within the cluster sorted by popularity
    return sorted_movies

from flask import request, session, redirect, url_for, flash, render_template

@app.route('/registration', methods=["GET", "POST"])
def registration():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']  # Assuming plaintext for simplicity, but should be hashed

        users_ref = db.collection('users')
        user_query = users_ref.where('email', '==', email).limit(1).get()

        if not user_query:  # No user found with the same email, so proceed with registration
            users_ref.add({'username': username, 'email': email, 'password': password})
            session['username'] = username  # Store username in session
            flash("Registration successful!")
            return redirect(url_for('display_movies_by_genre'))
        else:
            flash("Email already registered. Try logging in or use another email.")
            return redirect(url_for('registration'))

    return render_template('register.html')

@app.route('/signup')
def signup():
    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")
        
        users_ref = db.collection('users')
        user = users_ref.where('username', '==', username).where('password', '==', password).get()

        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html')
    else:
        return "<h1>Try another way</h1>"

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('display_movies_by_genre'))
df = pd.read_csv("top10k-TMDB-movies.csv")
def filter_movies_by_genre(df, genres, num_movies=5):
    filtered_movies = df[df['genre'].apply(lambda x: all(genre in x for genre in genres))]
    return filtered_movies.sample(min(num_movies, len(filtered_movies)))

@app.route('/')
def display_movies_by_genre():
    user_id = session.get('username', 'default_user')
    api_key='696bf4dc'
    
    # Load movies data
    user_data = pd.read_csv('ratings.csv', encoding='ISO-8859-1')
    new_movies = pd.read_csv('top10k-TMDB-movies.csv', encoding='ISO-8859-1')

    # Define genre combinations
    genre_combinations = {
        'Action Adventure': ['Action', 'Adventure'],
        'Drama Adventure': ['Drama', 'Adventure'],
        'Romance': ['Romance']
    }

    movies_by_genre = {}
    for genre_name, genres in genre_combinations.items():
        filtered_movies = filter_movies_by_genre(new_movies, genres, num_movies=5)
        movies_info = [{'title': row['title'], 'genre': row['genre']} for index, row in filtered_movies.iterrows()]
        movies_by_genre[genre_name] = movies_info

    return render_template('index.html', movies_by_genre=movies_by_genre, user=user_id)


def filter_movies_by_genre(df, genres, num_movies=5):
    filtered_movies = df[df['genre'].apply(lambda x: all(genre in x for genre in genres))]
    return filtered_movies.sample(min(num_movies, len(filtered_movies)))



def fetch_poster_url(title, api_key):
    search_url = f'http://www.omdbapi.com/?t={title}&apikey={api_key}'
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        poster_url = data.get('Poster')
        if poster_url and poster_url != 'N/A':
            return poster_url
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch poster for {title}. Error: {e}")
    return None
@app.route('/movie/<string:movie_title>')
def get_movie_info(movie_title):
    movie_title = movie_title.replace('-', ' ')
    api_key = '696bf4dc'
    movie_data = scrape_omdb_movie_data(movie_title, api_key)

    if movie_data:
        poster_url = fetch_poster_url(movie_title, api_key)
        movie_data['poster'] = poster_url  # Ensuring the poster URL is part of the movie data dictionary
        return render_template("movie_details.html", movie_data=movie_data)
    else:
        return f"Failed to retrieve movie information for '{movie_title}'"


    
def fetch_poster_url(title, api_key):
    search_url = f'http://www.omdbapi.com/?t={title}&apikey={api_key}'
    print("Fetching URL:", search_url)  # Debugging output
    try:
        response = requests.get(search_url)
        response.raise_for_status()  
        data = response.json()
        poster_url = data.get('Poster')
        print("Poster URL:", poster_url)  
        if poster_url and poster_url != 'N/A':
            return poster_url
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch poster for {title}. Error: {e}")
    return None




@app.route('/rate_movie/<string:movie_title>', methods=['POST', 'GET'])
def rate_movie(movie_title):
    if 'username' not in session:
        return jsonify({'error': "User not logged in"}), 401

    # Decode the movie title from URL encoding
    decoded_movie_title = unquote(movie_title)
    df = pd.read_csv('top10k-TMDB-movies.csv')
    movie = df.loc[df['title'] == decoded_movie_title]

    if movie.empty:
        return jsonify({'error': "Movie not found"})

    # Check for rating input
    rating = request.form.get('rating')
    if not rating:
        return jsonify({'error': "Rating not provided"})
    rating = int(rating)  # Ensure rating is an integer

    
    username = session['username']

    genres = movie.iloc[0]['genre']

    # Prepare to insert/update Firestore document
    db = firestore.client()
    ratings_ref = db.collection('UserRatings').document(f"{username}_{decoded_movie_title}")
    rating_data = {
        'username': username,
        'movie_title': decoded_movie_title,
        'rating': rating,
        'genre': genres
    }

    try:
        ratings_ref.set(rating_data)  # This will create or update the document
    except Exception as e:
        return jsonify({'error': f"Database error: {str(e)}"})

    return redirect(url_for('display_movies_by_genre'))
def generate_query_embedding(query_text):
    # Assuming an API key and function for generating embeddings
    response = openai.Embedding.create(
        input=[query_text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])


api_key='696bf4dc'
def find_similar_movies(query_text):
    query_embedding = generate_query_embedding(query_text)
    query_cluster = kmeans.predict(query_embedding.reshape(1, -1))[0]
    cluster_indices = np.where(clusters == query_cluster)[0]

    cluster_embeddings = embeddings_array[cluster_indices]
    local_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    local_model.fit(cluster_embeddings)
    distances, local_indices = local_model.kneighbors([query_embedding])

    similar_movies = [titles[cluster_indices[local_index]] for local_index in local_indices[0]]
    sorted_movies = sorted(similar_movies, key=lambda x: movie_popularity.get(x, 0), reverse=True)
    return sorted_movies

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('mood')
    similar_movies = find_similar_movies(query)
    api_key = 'sk-proj-xz1H9m4cUyBr0dKxdRHfT3BlbkFJKmD8yAHwNYun5ImeUrmm'
    detailed_movies = []
    for movie in similar_movies:
        movie_details = scrape_omdb_movie_data(movie, api_key)
        if movie_details:
            detailed_movies.append(movie_details)

    if detailed_movies:
        return render_template("recommendation.html", movies=detailed_movies)
    else:
        return f"No detailed movie data found for similar movies to '{query}'"

def scrape_omdb_movie_data(movie_title, api_key):
    search_url = f'https://www.omdbapi.com/?t={movie_title}&apikey={api_key}'
    try:
        response = requests.get(search_url)
        response.raise_for_status()  # Raises HTTPError for bad requests
        data = response.json()
        if data.get('Response') == 'True':
            poster_url = fetch_poster_url(movie_title, api_key) 
            return {
                'title': data.get('Title'),
                'overview': data.get('Plot'),
                'release_date': data.get('Released'),
                'rating': data.get('imdbRating'),
                'poster': poster_url or 'default_poster.jpg', 
                'director': data.get('Director'),
                'cast': data.get('Actors').split(', ')
            }
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {movie_title}. Error: {e}")
    return None


if __name__ == '__main__':
    app.run(debug=True)
