# 🎬 Movie Recommender System
<img width="1920" height="1688" alt="image" src="https://github.com/user-attachments/assets/76681b21-4cd6-45ed-9223-ea942f5d3205" />

An interactive web application built with Streamlit that provides personalized movie recommendations. This project leverages an **Item-Based Collaborative Filtering** model trained on the MovieLens dataset and enriches the user experience by fetching real-time movie data from the TMDb API.

---

## ✨ Features

- **Recommend by Movie:** Select a film you love and get a list of similar movies based on user rating patterns.
- **Recommend by Genre:** Choose your favorite genre to discover the most popular films within it.
- **Random Recommendations:** Feeling adventurous? Get a list of random movies with the "Surprise Me!" feature.
- **Interactive UI:** A clean, user-friendly interface powered by Streamlit.
- **Dynamic Movie Details:** Fetches and displays movie posters and synopses in real-time from the TMDb API.

---

## 🛠️ Tech Stack

- **Backend & ML Model:** Python, Pandas, Scikit-learn
- **Frontend:** Streamlit
- **API Integration:** TMDb API, Python `requests` library
- **Environment Management:** `python-dotenv` for secure API key handling

---

## 🚀 How to Run Locally

Follow these steps to get the application running on your local machine.

### Prerequisites

- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/bestoism/MovieRecommenderSystem.git
cd MovieRecommenderSystem
```

### 2. Create and Activate a Virtual Environment

**For Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (Crucial Step)

This project requires an API key from The Movie Database (TMDb).

1. Create a file named `.env` in the root directory of the project.
2. Add your TMDb API key to the `.env` file in the following format:
```
TMDB_API_KEY="YOUR_API_KEY_HERE"
```

3. **Note:** The `.env` file is included in `.gitignore` to ensure your API key is not pushed to GitHub.

### 5. Run the Streamlit Application

Once the setup is complete, run the following command in your terminal:
```bash
streamlit run app.py
```

The application will open automatically in your default web browser. The first time you run it, there will be a one-time process to compute and save the similarity model, which may take a few minutes. Subsequent runs will be instantaneous.

---

## 📂 Project Structure
```
MovieRecommenderSystem/
├── .gitignore
├── 01-Eksplorasi-Data.ipynb   # Jupyter Notebook for data exploration and model logic
├── app.py                     # Main Streamlit application file
├── data/                      # Raw MovieLens dataset files
├── requirements.txt           # Python dependencies
├── saved_model/               # (Generated locally, not in repo) Stores the computed similarity matrix
└── README.md
```

---

## 🙏 Acknowledgments

- This application uses the TMDb API but is not endorsed or certified by TMDb.
- The project is built upon the MovieLens (small) dataset.
