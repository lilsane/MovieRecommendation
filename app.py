from flask import Flask, render_template,request,url_for,redirect,jsonify,session,flash
import requests
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import random
import pandas as pd
import ast
import spacy
import random
import openai
import re
import csv
import pickle
from ast import literal_eval
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import mysql.connector
from urllib.parse import unquote


db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="sidharth",
    database="netflix"
)
mycursor = db.cursor()

app = Flask(__name__)
app.secret_key = 'a35fsw56uhg7ufjuh'



# Execute a query to fetch the data
mycursor.execute("SELECT * FROM UserRatings")

# Fetch all rows from the result set
rows = mycursor.fetchall()

# Close the cursor and connection


# Process the data
data = [['ID', 'Name', 'Movie', 'Rating','genre']]
for row in rows:
    data.append(row)
# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'sk-proj-xz1H9m4cUyBr0dKxdRHfT3BlbkFJKmD8yAHwNYun5ImeUrmm'

def generate_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-3-small"  # Using the specified model
    )
    # Extract the embeddings from the response data
    return [item['embedding'] for item in response['data']]

with open('ratings.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)

print("CSV file created successfully!")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
df=pd.read_csv('top10k-TMDB-movies.csv')
title_overview = df[['title', 'overview']]

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
    local_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    local_model.fit(cluster_embeddings)
    distances, local_indices = local_model.kneighbors([query_embedding])

    # Retrieve titles and sort by popularity
    similar_movies = [titles[cluster_indices[local_index]] for local_index in local_indices[0]]
    sorted_movies = sorted(similar_movies, key=lambda x: movie_popularity.get(x, 0), reverse=True)

    # Print or return the similar movies within the cluster sorted by popularity
    return sorted_movies

# Example Usage
find_similar_movies("spiderman")

#USER IDENTIFICATION
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")
        query = "SELECT * from register where username=%s and password=%s"
        data = (username, password)
        mycursor.execute(query, data)
        result = mycursor.fetchall()
        if result:
            global user_active, email_id
            user_active = str(result[0][0])
            email_id = username
            session['username'] = username
            return redirect(url_for('home'))
            # return render_template('dashboard.html',username=username)
        else:
            return render_template('login.html')
    else:
        return "<h1>try another way</h1>"
    
@app.route('/register', methods=["GET", "POST"])
def registration():
    if request.method == "POST":
        username=request.form['username']
        
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        session['username'] = username
        mycursor.execute("SELECT * FROM register WHERE email = %s", (email,))
        user_exist = mycursor.fetchone()

        if user_exist:
            flash("The email is already registered. Try another email.")
        else:
            mycursor.execute("INSERT INTO register(username, email, password) VALUES (%s, %s, %s)",
                             (username,email,password))
            db.commit()
            flash(f"Registration successful for {username}")

    return redirect(url_for('display_movies_by_genre'))


@app.route('/registeration')
def register():
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('display_movies_by_genre'))

df = pd.read_csv("top10k-TMDB-movies.csv")
df['genre'] = df['genre'].apply(ast.literal_eval)

def fetch_poster_url(title, api_key):
    search_url = f'https://api.themoviedb.org/3/search/movie?query={title}&api_key={api_key}'
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results')
        if results and len(results) > 0:
            # Assuming the first result is the most relevant
            poster_path = results[0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None



def filter_movies_by_genre(genres, num_movies=5):
    filtered_movies = df[df['genre'].apply(lambda x: all(genre in x for genre in genres))]
    return filtered_movies.sample(min(num_movies, len(filtered_movies)))


@app.route('/')
def display_movies_by_genre():
    user_id = session.get('username', 'default_user')
    api_key='65aef011fa777723dbe0829346b0f519'
    
    # Load movies data
    user_data = pd.read_csv('ratings.csv', encoding='ISO-8859-1')
    new_movies = pd.read_csv('top10k-TMDB-movies.csv', encoding='ISO-8859-1')

    # Define genre combinations
    genre_combinations = {
        'Action Adventure': ['Action', 'Adventure'],
        'Drama Adventure': ['Drama', 'Adventure'],
        'Romance': ['Romance'],
        # Add more as needed
    }

    # Fetching movies by genres
    movies_by_genre = {}
    for genre_name, genres in genre_combinations.items():
        sampled_movies = filter_movies_by_genre(genres, num_movies=5)
        movies_info = []
        for _, movie in sampled_movies.iterrows():
            movie_dict = movie.to_dict()
            movie_dict['poster'] = fetch_poster_url(movie['title'],api_key)
            movies_info.append(movie_dict)
        movies_by_genre[genre_name] = movies_info

    recommendations = []
    if user_id and user_id != 'default_user':
        # Filtering and processing for personalized recommendations
        current_user_ratings = user_data[user_data['Name'] == user_id]
        current_user_ratings['genre'] = current_user_ratings['genre'].apply(literal_eval)
        liked_genres = current_user_ratings[current_user_ratings['Rating'] > 3]['genre'].explode().value_counts()

        new_movies['genre'] = new_movies['genre'].apply(literal_eval)
        new_movies['match_score'] = new_movies['genre'].apply(lambda genres: sum(genre in liked_genres.index for genre in genres))
        recommended_movies = new_movies.sort_values(by='match_score', ascending=False).head(5)
        recommendations = recommended_movies[['title', 'genre']].to_dict(orient='records')

    # Render a single template with both recommendations and genre-based movies
    return render_template('index.html', movies_by_genre=movies_by_genre, recommendations=recommendations, user=user_id)

def scrape_tmdb_movie_data(movie_title, api_key):
    # Search for the movie by title to get its ID
    search_url = f'https://api.themoviedb.org/3/search/movie?query={movie_title}&api_key={api_key}'
    search_response = requests.get(search_url)
    
    if search_response.status_code == 200:
        search_data = search_response.json()
        results = search_data.get('results')
        
        if results and len(results) > 0:
            # Get the ID of the first result to fetch detailed movie info
            movie_id = results[0].get('id')
            
            # Use the ID to request detailed movie information, including cast and crew
            details_url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits'
            details_response = requests.get(details_url)
            
            if details_response.status_code == 200:
                details_data = details_response.json()
                
                # Extract director and cast
                crew = details_data.get('credits', {}).get('crew', [])
                cast = details_data.get('credits', {}).get('cast', [])
                
                director = next((member['name'] for member in crew if member['job'] == 'Director'), None)
                top_cast = [actor['name'] for actor in cast[:5]]  # Get names of top 5 cast members
                
                movie_details = {
                    'title': details_data.get('title'),
                    'overview': details_data.get('overview'),
                    'release_date': details_data.get('release_date'),
                    'rating': details_data.get('vote_average'),
                    'poster': fetch_poster_url(details_data.get('poster_path'), api_key),
                    'director': director,
                    'cast': top_cast
                }
                return movie_details
            else:
                print(f"Failed to retrieve detailed data from TMDB API. Status code: {details_response.status_code}")
                return None
        else:
            print(f"No results found for the movie '{movie_title}'.")
            return None
    else:
        print(f"Failed to retrieve data from TMDB API. Status code: {search_response.status_code}")
        return None

@app.route('/movie/<string:movie_title>')
def get_movie_info(movie_title):
    
    movie_title = movie_title.replace('-', ' ')

    
    api_key = 'bafc1925'

   
    movie_data = scrape_tmdb_movie_data(movie_title, api_key)

    if movie_data:
        return render_template("movie_details.html",movie_data=movie_data)
    else:
        return f"Failed to retrieve movie information for '{movie_title}'"


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
    rating = int(rating)  # Ensure rating is a native Python int

    # Insert into database
    username = session['username']  # Retrieve username from session
    genres = movie.iloc[0]['genre']  # Assume 'genre' is a column in the DataFrame

    # Insert statement now uses movie_title instead of movie_id
    sql = "INSERT INTO UserRatings (username, movie_title, rating, genre) VALUES (%s, %s, %s, %s)"
    values = (username, decoded_movie_title, rating, genres)
    try:
        mycursor.execute(sql, values)
        db.commit()
    except mysql.connector.Error as e:
        return jsonify({'error': "Database error: " + str(e)})

    return redirect(url_for('display_movies_by_genre'))




@app.route('/search', methods=['GET'])
def search():
    query=request.args.get('mood')
    similar_movies=find_similar_movies(query)
    if not similar_movies.empty:
        api_key='bafc1925'
        detailed_movies=[]
        for index, row in similar_movies.iterrows():
            movie_details=scrape_tmdb_movie_data(row['title'],api_key)
            if movie_details:
                detailed_movies.append(movie_details)
        if detailed_movies:  
            return render_template("recommandation.html", movies=detailed_movies)
        else:
            return f"No detailed movie data found for similar movies to '{query}'"
    else:
        return f"No similar movies found for '{query}'"

if __name__ == '__main__':
    app.run(debug=True)
