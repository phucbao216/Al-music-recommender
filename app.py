import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Streamify AI", page_icon="🎵", layout="wide")

# --- 2. SPOTIFY API SETUP ---
# Make sure these are EXACTLY as in your Dashboard
CLIENT_ID = '19c97c85e06d4f4882183efcea8615c9'
CLIENT_SECRET = 'ae0fd04a4a5b4b22a4d90ad6564c11d4'

try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except Exception as e:
    st.error("Spotify API Connection Failed. Check your Credentials.")

def get_spotify_data(track_name, artist_name):
    """Refined search: Only take the main song title before '-' or '(' """
    try:
        # Clean song name: "Shape of You - Remastered" -> "Shape of You"
        clean_name = track_name.split('-')[0].split('(')[0].strip()
        query = f"track:{clean_name} artist:{artist_name}"
        
        results = sp.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            # Get the high-resolution cover
            album_cover = track['album']['images'][0]['url'] if track['album']['images'] else None
            # Get the preview URL
            preview_url = track['preview_url']
            return album_cover, preview_url
    except Exception as e:
        # This will print the error to your Streamlit Logs so we can debug easier
        print(f"Spotify API Error: {e}")
    return None, None

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop_duplicates(subset=['track_name', 'artists']).dropna()
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features

df, features = load_data()

# --- 4. ENGINE ---
def recommend_songs(song_name, num_recommendations=5):
    # Fuzzy search for the song in our local dataset
    search = df[df['track_name'].str.contains(song_name, case=False)]
    if search.empty: return None
    
    target = search.iloc[0]
    target_vec = target[features].values.reshape(1, -1)
    
    # Filter by genre to ensure the "vibe" is correct
    candidates = df[df['track_genre'] == target['track_genre']].copy()
    candidates = candidates[candidates['track_name'] != target['track_name']]
    
    if candidates.empty: candidates = df.copy()

    sim = cosine_similarity(target_vec, candidates[features])
    candidates['score'] = (sim[0] * 0.8) + (candidates['popularity'] / 100.0 * 0.2)
    return candidates.sort_values(by='score', ascending=False).head(num_recommendations)

# --- 5. UI ---
st.title("🎵 Streamify AI")
st.markdown("Global music discovery powered by **Machine Learning**.")

user_input = st.text_input("🔍 Search for a song:", placeholder="Try 'Starboy' or 'Shape of You'...")

if user_input:
    results = recommend_songs(user_input)
    if results is not None:
        st.success("Recommended Tracks:")
        for _, row in results.iterrows():
            cover, preview = get_spotify_data(row['track_name'], row['artists'])
            
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    if cover: st.image(cover)
                    else: st.markdown("### 💿")
                with c2:
                    st.subheader(row['track_name'])
                    st.write(f"**Artist:** {row['artists']} | **Genre:** {row['track_genre'].title()}")
                    if preview: st.audio(preview)
                    else: st.caption("Preview audio not available on Spotify")
                st.divider()
    else:
        st.warning("Song not found. Try another one!")
