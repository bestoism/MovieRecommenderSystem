# app.py (Versi Final dengan Perbaikan NameError)

import streamlit as st
import pandas as pd
import random
import requests
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity # <-- Import di atas

# Memuat variabel dari file .env
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# Pengecekan API Key di awal
if not API_KEY:
    st.error("API Key TMDb tidak ditemukan. Mohon buat file .env dan tambahkan TMDB_API_KEY.")
    st.stop()

# --- 1. FUNGSI-FUNGSI UTAMA ---

# DEFINISIKAN load_data() TERLEBIH DAHULU
@st.cache_data
def load_data():
    """Memuat semua data yang sudah diproses dari folder saved_model."""
    movies = pd.read_csv('saved_model/movies_cleaned.csv')
    item_similarity_df = pd.read_csv('saved_model/item_similarity.csv', index_col=0)
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    links = pd.read_csv('data/ml-latest-small/links.csv')
    
    movies_with_ids = pd.merge(movies, links, on='movieId')
    movies_with_ids = movies_with_ids.dropna(subset=['tmdbId'])
    movies_with_ids['tmdbId'] = movies_with_ids['tmdbId'].astype(int)
    
    return movies_with_ids, item_similarity_df, ratings

@st.cache_resource # Gunakan cache_resource untuk komputasi berat
def prepare_data_and_model():
    """
    Memeriksa file model. Jika tidak ada, jalankan komputasi dan simpan.
    Setelah itu, panggil load_data() untuk memuat dan mengembalikan dataframes.
    """
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    item_sim_path = 'saved_model/item_similarity.csv'
    movies_clean_path = 'saved_model/movies_cleaned.csv'

    if not os.path.exists(item_sim_path):
        with st.spinner("Persiapan awal sedang dilakukan... Ini mungkin butuh beberapa menit dan hanya terjadi sekali."):
            ratings_raw = pd.read_csv('data/ml-latest-small/ratings.csv')
            movies_raw = pd.read_csv('data/ml-latest-small/movies.csv')
            
            movies_raw.to_csv(movies_clean_path, index=False)
            
            # Komputasi model
            user_item_matrix = ratings_raw.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
            item_user_matrix = user_item_matrix.T
            item_similarity = cosine_similarity(item_user_matrix)
            item_similarity_df = pd.DataFrame(item_similarity,
                                              index=item_user_matrix.index,
                                              columns=item_user_matrix.index)
            item_similarity_df.to_csv(item_sim_path)
            # st.success("Persiapan selesai! Model telah dibuat dan disimpan secara lokal.") # Bisa di-uncomment jika perlu
    
    # Setelah memastikan file ada, muat datanya
    return load_data()

# ... (Fungsi-fungsi helper lainnya tetap sama) ...
def get_item_recommendations(movie_title, item_similarity_df, movies_df, n=10):
    if movie_title in item_similarity_df.index:
        sim_scores = item_similarity_df[movie_title].sort_values(ascending=False)
        top_movies_indices = sim_scores.iloc[1:n+1].index
        return movies_df[movies_df['title'].isin(top_movies_indices)]
    return pd.DataFrame()

def get_genre_recommendations(genre, movies_df, ratings_df, n=10):
    genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]
    if not genre_movies.empty:
        movie_ratings = ratings_df[ratings_df['movieId'].isin(genre_movies['movieId'])]
        popularity = movie_ratings['movieId'].value_counts().reset_index()
        popularity.columns = ['movieId', 'rating_count']
        popular_movies = pd.merge(genre_movies, popularity, on='movieId').sort_values('rating_count', ascending=False)
        return popular_movies.head(n)
    return pd.DataFrame()

def get_random_recommendations(movies_df, n=10):
    return movies_df.sample(n)

@st.cache_data
def fetch_movie_details(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        full_poster_path = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/500x750.png?text=No+Poster"
        overview = data.get('overview', 'No synopsis available.')
        return full_poster_path, overview
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750.png?text=Error", "Could not fetch details."

# --- 2. INISIALISASI APLIKASI ---

# Panggil fungsi persiapan SATU KALI untuk mendapatkan semua data
movies_df, item_similarity_df, ratings_df = prepare_data_and_model()

# Siapkan list untuk UI
movie_list = [""] + sorted(movies_df['title'].unique().tolist())
genre_list = [""] + sorted(list(set('|'.join(movies_df['genres']).split('|'))))

# Inisialisasi session state
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'recommendations' not in st.session_state: st.session_state.recommendations = pd.DataFrame()
if 'selected_movie' not in st.session_state: st.session_state.selected_movie = None

# --- 3. LOGIKA TAMPILAN (UI PAGES) ---
# (Semua fungsi display_... TIDAK BERUBAH, tapi sekarang mereka akan menggunakan dataframes global yang sudah benar)

def display_home_page():
    st.title("🎬 Movie Recommender")
    st.write("Temukan film favoritmu berikutnya! Pilih berdasarkan film yang kamu suka atau genre favoritmu.")
    col1, col2, col3 = st.columns([3, 3, 1])
    with col1: selected_movie = st.selectbox("Ketik atau pilih film yang kamu suka:", movie_list)
    with col2: selected_genre = st.selectbox("Atau pilih genre favoritmu:", genre_list)
    with col3:
        st.write(""); st.write("")
        if st.button("🎲 Acak!", help="Tampilkan film acak"):
            with st.spinner('Mencari film seru...'):
                st.session_state.recommendations = get_random_recommendations(movies_df)
                st.session_state.page = 'recommendations'
                st.rerun()
    if st.button("Tampilkan Rekomendasi", type="primary"):
        with st.spinner('Mencari rekomendasi untukmu...'):
            if selected_movie: st.session_state.recommendations = get_item_recommendations(selected_movie, item_similarity_df, movies_df)
            elif selected_genre: st.session_state.recommendations = get_genre_recommendations(selected_genre, movies_df, ratings_df)
            else:
                st.warning("Silakan pilih film atau genre terlebih dahulu.")
                return
            if not st.session_state.recommendations.empty:
                st.session_state.page = 'recommendations'
                st.rerun()
            else: st.error("Maaf, tidak dapat menemukan rekomendasi. Coba film atau genre lain.")

def display_recommendations_page():
    st.title("Berikut Rekomendasi Untukmu")
    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.page = 'home'
        st.rerun()
    recs = st.session_state.recommendations
    num_cols = 5
    cols = st.columns(num_cols)
    for i, row in recs.reset_index().iterrows():
        with cols[i % num_cols]:
            poster_url, _ = fetch_movie_details(row['tmdbId'])
            st.image(poster_url, use_container_width=True)
            st.write(f"**{row['title']}**") 
            st.caption(f"{row['genres']}")
            if st.button("Lihat Detail", key=f"detail_{row['movieId']}"):
                st.session_state.selected_movie = row.to_dict()
                st.session_state.page = 'details'
                st.rerun()

def display_movie_details_page():
    movie = st.session_state.selected_movie
    st.title(movie['title'])
    poster_url, overview = fetch_movie_details(movie['tmdbId'])
    if st.button("⬅️ Kembali ke Rekomendasi"):
        st.session_state.page = 'recommendations'
        st.rerun()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(poster_url, use_container_width=True)
    with col2:
        st.subheader("Genre")
        st.write(movie['genres'].replace('|', ', '))
        st.subheader("Rating Rata-rata (dari data)")
        avg_rating = ratings_df[ratings_df['movieId'] == movie['movieId']]['rating'].mean()
        st.write(f"⭐ {avg_rating:.1f} / 5.0" if not pd.isna(avg_rating) else "Belum ada rating")
        st.subheader("Sinopsis")
        st.write(overview)
        if st.button("Rekomendasikan Film Mirip Ini", type="primary"):
             with st.spinner('Mencari film yang mirip...'):
                st.session_state.recommendations = get_item_recommendations(movie['title'], item_similarity_df, movies_df)
                st.session_state.page = 'recommendations'
                st.rerun()

# --- 4. ROUTER UTAMA ---
if st.session_state.page == 'home': display_home_page()
elif st.session_state.page == 'recommendations': display_recommendations_page()
elif st.session_state.page == 'details': display_movie_details_page()