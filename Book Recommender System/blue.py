import pickle
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px

# Streamlit page setup
st.set_page_config(page_title="Book Recommender Dashboard", layout="wide")
st.header("📚 Book Recommender System using PySpark")

# Load model and data
model = pickle.load(open(r'C:/Users/Manamnath tiwari/OneDrive/Desktop/DataScience/Projects/Recommender System/Projects/Book Recommender System/artificats/model.pkl', 'rb'))
book_pivot = pickle.load(open(r'C:/Users/Manamnath tiwari/OneDrive/Desktop/DataScience/Projects/Recommender System/Projects/Book Recommender System/artificats/book_pivot.pkl', 'rb'))
books_name = pickle.load(open(r'C:/Users/Manamnath tiwari/OneDrive/Desktop/DataScience/Projects/Recommender System/Projects/Book Recommender System/artificats/books_name.pkl', 'rb'))
final_rating = pickle.load(open(r'C:/Users/Manamnath tiwari/OneDrive/Desktop/DataScience/Projects/Recommender System/Projects/Book Recommender System/artificats/final_rating.pkl', 'rb'))

# -------------------------------------------
# 📌 RECOMMENDATION SYSTEM
# -------------------------------------------
def fetch_poster(suggestion):
    book_names = []
    ids_index = []
    poster_urls = []

    for book_id in suggestion[0]:
        book_names.append(book_pivot.index[book_id])

    for name in book_names:
        ids = np.where(final_rating['title'] == name)[0]
        if len(ids) > 0:
            ids_index.append(ids[0])
    
    for ids in ids_index:
        poster_urls.append(final_rating.iloc[ids]['img_url'])
    
    return poster_urls

def recommended_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    poster_urls = fetch_poster(suggestion)
    for i in suggestion[0]:
        book_list.append(book_pivot.index[i])
    
    return book_list, poster_urls, suggestion[0]

# -------------------------------------------
# 📊 VISUALIZATIONS
# -------------------------------------------
with st.expander("📊 Click to View Visualizations"):
    st.subheader("Top 20 Most Rated Books")
    top_books = final_rating['title'].value_counts().head(20)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.barplot(y=top_books.index, x=top_books.values, palette='mako', ax=ax1)
    ax1.set_xlabel("Number of Ratings")
    ax1.set_ylabel("Book Title")
    st.pyplot(fig1)

    st.subheader("Rating Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='rating', data=final_rating, ax=ax2)
    ax2.set_title("Distribution of Book Ratings")
    st.pyplot(fig2)

    st.subheader("Top 15 Authors by Number of Books")
    top_authors = final_rating['author'].value_counts().head(15)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_authors.index, x=top_authors.values, palette='viridis', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Top 10 Publishers by Number of Books")
    top_publishers = final_rating['publisher'].value_counts().head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_publishers.index, x=top_publishers.values, palette='cubehelix', ax=ax4)
    st.pyplot(fig4)

    st.subheader("Number of Books Published Over the Years")
    year_data = pd.to_numeric(final_rating['year'], errors='coerce')
    year_data = year_data[(year_data > 1900) & (year_data < 2025)]
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.histplot(year_data, bins=30, kde=True, ax=ax5, color='green')
    st.pyplot(fig5)

# -------------------------------------------
# 🔍 USER INPUT + RECOMMENDATIONS
# -------------------------------------------
selected_books = st.selectbox("📘 Type or Select a Book", books_name)

if st.button('🔎 Show Recommendations'):
    recommendation_books, poster_urls, suggested_indices = recommended_books(selected_books)
    st.subheader("🔁 Top 5 Book Recommendations")
    col_list = st.columns(5)
    for i, col in enumerate(col_list):
        if i < len(recommendation_books):
            with col:
                st.text(recommendation_books[i])
                st.image(poster_urls[i])

    # -------------------------------------------
    # 📈 3D INTERACTIVE PLOT (Plotly)
    # -------------------------------------------
    st.subheader("📌 3D Visualization of Book Recommendations")

    # Reduce book_pivot to 3D
    pca = PCA(n_components=3)
    book_pivot_3d = pca.fit_transform(book_pivot.values)

    # Create DataFrame for Plotly
    pca_df = pd.DataFrame(book_pivot_3d, columns=['x', 'y', 'z'])
    pca_df['title'] = book_pivot.index

    # Mark selected and recommended books
    pca_df['category'] = 'Other'
    pca_df.loc[pca_df['title'] == selected_books, 'category'] = 'Selected Book'
    for idx in suggested_indices:
        pca_df.loc[pca_df['title'] == book_pivot.index[idx], 'category'] = 'Recommended Book'

    fig = px.scatter_3d(pca_df, x='x', y='y', z='z', color='category', hover_name='title',
                        color_discrete_map={'Selected Book': 'red', 'Recommended Book': 'blue', 'Other': 'gray'},
                        size_max=18)
    fig.update_layout(height=700, title="3D Projection of Book Similarities (PCA)")
    st.plotly_chart(fig, use_container_width=True)
