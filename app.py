import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Streamify AI", page_icon="🎵", layout="wide")

# --- 2. SPOTIFY API SETUP ---
# Replace with your actual credentials from Spotify Developer Dashboard
CLIENT_ID = 'YOUR_CLIENT_ID_HERE'
CLIENT_SECRET = 'YOUR_CLIENT_SECRET_HERE'

# Initialize Spotify client
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_data(track_name, artist_name):
    """
    Fetches album cover URL and a 30-second audio preview from Spotify API.
    """
    try:
        query = f"track:{track_name} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            album_cover = track['album']['images'][0]['url'] if track['album']['images'] else None
            preview_url = track['preview_url']
            return album_cover, preview_url
    except Exception as e:
        print(f"Error fetching Spotify data: {e}")
    return None, None

# --- 3. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    df = df.dropna()

    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features

df, features = load_data()

# --- 4. RECOMMENDATION LOGIC ---
def recommend_songs(song_name, num_recommendations=5):
    search_results = df[df['track_name'].str.contains(song_name, case=False)]
    if search_results.empty:
        return None

    song_index = search_results.index[0]
    target_song = df.loc[song_index]
    target_vector = target_song[features].values.reshape(1, -1)
    target_genre = target_song['track_genre']

    # Filtering by genre and calculating similarity
    candidate_pool = df[df['track_genre'] == target_genre].copy()
    candidate_pool = candidate_pool[candidate_pool.index != song_index]
    
    if candidate_pool.empty:
        candidate_pool = df[df.index != song_index].copy()

    similarity_scores = cosine_similarity(target_vector, candidate_pool[features])
    candidate_pool['similarity'] = similarity_scores[0]
    candidate_pool['norm_popularity'] = candidate_pool['popularity'] / 100.0

    # Hybrid Score: 70% Similarity + 30% Popularity
    candidate_pool['hybrid_score'] = (candidate_pool['similarity'] * 0.7) + (candidate_pool['norm_popularity'] * 0.3)
    
    final_recs = candidate_pool.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)
    return final_recs

# --- 5. USER INTERFACE ---
st.title("🎵 Streamify AI")
st.markdown("Experience intelligent music discovery powered by **Machine Learning** and **Spotify API**.")

st.divider()
user_input = st.text_input("🔍 Search for a song to find similar vibes:", placeholder="e.g., Blinding Lights, My Love...")

if user_input:
    with st.spinner("Analyzing the rhythm..."):
        results = recommend_songs(user_input)

    if results is not None:
        st.success(f"Found matches for '{user_input.title()}'!")
        
        # Displaying recommendations in an elegant grid layout
        st.subheader("Top Recommended Tracks")
        
        for _, row in results.iterrows():
            song_title = row['track_name']
            artist_name = row['artists']
            
            # Fetch real-time data from Spotify
            cover, preview = get_spotify_data(song_title, artist_name)
            
            # Create a container for each song
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if cover:
                        st.image(cover, use_container_width=True)
                    else:
                        st.write("💿 No Cover")
                
                with col2:
                    st.markdown(f"### {song_title}")
                    st.write(f"**Artist:** {artist_name} | **Genre:** {row['track_genre'].title()}")
                    
                    if preview:
                        st.audio(preview, format="audio/mp3")
                    else:
                        st.info("💡 Preview not available on Spotify for this track.")
                
                st.divider()
    else:
        st.error("Song not found. Please try another one!")
