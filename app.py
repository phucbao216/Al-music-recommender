import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# --- Page Configuration ---
st.set_page_config(page_title="Streamify AI", page_icon="🎵")


# --- 1. Data Loading & Caching ---
# @st.cache_data is a Streamlit superpower. It saves the processed data in RAM
# so the app doesn't have to reload the massive CSV file every time the user types a letter.
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    df = df.dropna()

    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features


# Load the data into the app
df, features = load_data()


# --- 2. HYBRID Recommendation Logic ---
def recommend_songs(song_name, num_recommendations=5):
    """
    Hybrid Recommendation Engine: Combines Audio Similarity + Genre Locking + Popularity.
    """
    # Find the target song
    search_results = df[df['track_name'].str.contains(song_name, case=False)]
    if search_results.empty:
        return None

    # Extract target song details
    song_index = search_results.index[0]
    target_song = df.loc[song_index]
    target_vector = target_song[features].values.reshape(1, -1)
    target_genre = target_song['track_genre']

    # STEP 1: GENRE LOCKING
    # Filter songs that share the exact same genre to maintain the "vibe"
    candidate_pool = df[df['track_genre'] == target_genre].copy()
    candidate_pool = candidate_pool[candidate_pool.index != song_index]

    # Fallback if genre pool is unexpectedly empty
    if candidate_pool.empty:
        candidate_pool = df[df.index != song_index].copy()

    # STEP 2: AUDIO SIMILARITY
    # Calculate how mathematically similar the songs are (Scale: 0 to 1)
    similarity_scores = cosine_similarity(target_vector, candidate_pool[features])
    candidate_pool['similarity'] = similarity_scores[0]

    # STEP 3: THE HYBRID SCORING SYSTEM
    # Normalize popularity (0-100) to match similarity scale (0-1)
    candidate_pool['norm_popularity'] = candidate_pool['popularity'] / 100.0

    # Create a combined score: 70% Audio Match + 30% Crowd Popularity
    # You can tweak these weights later!
    weight_audio = 0.7
    weight_popularity = 0.3

    candidate_pool['hybrid_score'] = (candidate_pool['similarity'] * weight_audio) + (
                candidate_pool['norm_popularity'] * weight_popularity)

    # Sort by the final hybrid score and get Top N
    final_recommendations = candidate_pool.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)

    columns_to_return = ['track_name', 'artists', 'track_genre', 'popularity'] + features
    return final_recommendations[columns_to_return]


# --- 3. Visualization Logic ---
def get_visualization(song_name, recommendations):
    """
    Instead of plt.show(), we return the 'fig' object to render it on the webpage.
    """
    df_sample = df.sample(5000, random_state=42)
    target_song = df[df['track_name'].str.contains(song_name, case=False)].head(1)

    plot_df = pd.concat([df_sample, target_song, recommendations])

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(plot_df[features])

    plot_df['pca_x'] = pca_results[:, 0]
    plot_df['pca_y'] = pca_results[:, 1]

    # Create the figure object explicitly
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_len = len(df_sample)
    target_len = len(target_song)

    background_data = plot_df.iloc[:sample_len]
    target_data = plot_df.iloc[sample_len: sample_len + target_len]
    rec_data = plot_df.iloc[sample_len + target_len:]

    sns.scatterplot(data=background_data, x='pca_x', y='pca_y', color='lightgray', alpha=0.5, label='Other Songs',
                    ax=ax)
    sns.scatterplot(data=rec_data, x='pca_x', y='pca_y', color='blue', s=100, label='Recommendations', ax=ax)
    sns.scatterplot(data=target_data, x='pca_x', y='pca_y', color='red', s=200, marker='*', label='Target Song', ax=ax)

    ax.set_title(f"Music Galaxy for '{song_name.title()}'")

    # Return the figure to Streamlit
    return fig


# --- 4. Streamlit User Interface (UI) ---
st.title("🎵 Streamify AI")
st.markdown("Discover new tracks based on acoustic similarity. Powered by **Machine Learning**.")

# Search bar layout
st.divider()
user_input = st.text_input("🔍 Search for a song you love:", placeholder="e.g., Shape of You, Blinding Lights...")

# Only run if the user has typed something
if user_input:
    with st.spinner("Searching the acoustic galaxy..."):
        results = recommend_songs(user_input)

    if results is not None:
        st.success("Found some great matches!")

        # Display the table cleanly
        st.subheader(f"Top 5 Similar Tracks")
        display_df = results[['track_name', 'artists', 'track_genre', 'popularity']]
        # Streamlit interactive dataframe
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Display the chart
        st.subheader("🌌 The Music Galaxy")
        st.markdown("See how your song (⭐) clusters with the recommendations (🔵) across 5000 random tracks.")
        fig = get_visualization(user_input, results)
        st.pyplot(fig)
    else:
        st.error(f"Oops! Couldn't find '{user_input}' in our 114k song database. Try another one!")
