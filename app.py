import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Streamify AI", page_icon="🎵", layout="wide")

# --- 2. SPOTIFY API SETUP ---
# Integrated your generated credentials
CLIENT_ID = '19c97c85e06d4f4882183efcea8615c9'
CLIENT_SECRET = 'ae0fd04a4a5b4b22a4d90ad6564c11d4'

# Initialize Spotify connection
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_data(track_name, artist_name):
    """Fetches album cover and audio preview from Spotify API."""
    try:
        query = f"track:{track_name} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            album_cover = track['album']['images'][0]['url'] if track['album']['images'] else None
            preview_url = track['preview_url']
            return album_cover, preview_url
    except Exception:
        return None, None

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop_duplicates(subset=['track_name', 'artists']).dropna()
    
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features

df, features = load_data()

# --- 4. RECOMMENDATION ENGINE ---
def recommend_songs(song_name, num_recommendations=5):
    search_results = df[df['track_name'].str.contains(song_name, case=False)]
    if search_results.empty:
        return None

    target_song = search_results.iloc[0]
    target_vector = target_song[features].values.reshape(1, -1)
    
    # Filter by same genre for better accuracy
    candidate_pool = df[df['track_genre'] == target_song['track_genre']].copy()
    candidate_pool = candidate_pool[candidate_pool['track_name'] != target_song['track_name']]
    
    if candidate_pool.empty:
        candidate_pool = df.copy()

    # Calculate similarity & combine with popularity
    similarity = cosine_similarity(target_vector, candidate_pool[features])
    candidate_pool['score'] = (similarity[0] * 0.7) + (candidate_pool['popularity'] / 100.0 * 0.3)
    
    return candidate_pool.sort_values(by='score', ascending=False).head(num_recommendations)

# --- 5. MODERN UI ---
st.title("🎵 Streamify AI")
st.markdown("Discover your next favorite track using **Machine Learning** & **Spotify Data**.")

user_input = st.text_input("🔍 Search for a song you love:", placeholder="e.g., Starboy, Perfect...")

if user_input:
    with st.spinner("Finding the perfect match..."):
        results = recommend_songs(user_input)

    if results is not None:
        st.success(f"Recommended for you:")
        for _, row in results.iterrows():
            cover, preview = get_spotify_data(row['track_name'], row['artists'])
            
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    if cover: st.image(cover)
                    else: st.write("💿")
                with col2:
                    st.markdown(f"### {row['track_name']}")
                    st.write(f"**Artist:** {row['artists']} | **Genre:** {row['track_genre'].title()}")
                    if preview: st.audio(preview)
                    else: st.caption("Preview unavailable")
                st.divider()
    else:
        st.error("Song not found in our database. Try another one!")
