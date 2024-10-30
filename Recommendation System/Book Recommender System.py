import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import messagebox

class BookRecommenderSystem:
    def __init__(self, books, ratings):
        self.books_df = pd.DataFrame(books)
        self.ratings_df = pd.DataFrame(ratings)
        
        self.merged_df = pd.merge(self.ratings_df, self.books_df, on='book_id')
        
        self.ratings_matrix = self.merged_df.pivot(index='user_id', columns='title', values='rating').fillna(0)
        
        self.ratings_matrix_np = self.ratings_matrix.values
        
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.ratings_matrix_np)

    def find_similar_users(self, user_id, k=3):
        """Find similar users based on their book ratings using k-NN."""
        distances, indices = self.model_knn.kneighbors(self.ratings_matrix.loc[user_id, :].values.reshape(1, -1), n_neighbors=k+1)
        similar_users = list(indices.flatten())[1:]
        return similar_users

    def recommend_books(self, user_id, num_recommendations=5):
        """Recommend books to a user based on the ratings of similar users."""
        try:
            similar_users = self.find_similar_users(user_id, k=3)
            similar_users_ratings = self.ratings_matrix.iloc[similar_users].mean(axis=0)
            user_ratings = self.ratings_matrix.loc[user_id]
            books_not_rated = user_ratings[user_ratings == 0]
            recommendations = similar_users_ratings[books_not_rated.index].sort_values(ascending=False).head(num_recommendations)
            return recommendations
        except KeyError:
            return None

    def get_author(self, title):
        """Get the author of a given book title."""
        book_row = self.books_df[self.books_df['title'] == title]
        if not book_row.empty:
            return book_row.iloc[0]['author']
        return "Unknown Author"


class BookRecommenderGUI:
    def __init__(self, recommender_system):
        self.recommender_system = recommender_system
        
        self.window = tk.Tk()
        self.window.title("Book Recommender System")
        self.window.geometry('600x450')

        self.user_label = tk.Label(self.window, text="Enter User ID:", font=('Arial', 20))
        self.user_label.pack(pady=15)
        
        self.user_id_entry = tk.Entry(self.window, font=('Arial', 20))
        self.user_id_entry.pack(pady=7)
        
        self.recommend_button = tk.Button(self.window, text="Recommend Books", command=self.generate_recommendations, font=('Arial', 15))
        self.recommend_button.pack(pady=15)
        
        self.result_area = tk.Text(self.window, height=15, width=50, font=('Arial', 15))
        self.result_area.pack(pady=15)

        self.window.mainloop()

    def generate_recommendations(self):
        """Generates book recommendations when button is clicked."""
        user_id = self.user_id_entry.get()

        if user_id.isdigit():
            user_id = int(user_id)
            recommendations = self.recommender_system.recommend_books(user_id)

            if recommendations is not None and not recommendations.empty:
                self.result_area.delete(1.0, tk.END)
                self.result_area.insert(tk.END, "Recommended Books:\n\n")
                for title, rating in recommendations.items():
                    author = self.recommender_system.get_author(title)
                    self.result_area.insert(tk.END, f"Title: {title}\nAuthor: {author}\nPredicted Rating: {rating:.2f}\n\n")
            else:
                messagebox.showerror("Error", "User not found or no recommendations available.")
        else:
            messagebox.showwarning("Input Error", "Please enter a valid numeric User ID.")

books = {
    'book_id': [1, 2, 3, 4, 5],
    'title': ['Harry Potter and the Prisoner of Azkaban', 'The Alchemist', 'The Da Vinci Code', 'Gone with the Wind', 'Pride and Prejudice'],
    'author': ['J.K Rowling', 'Paulo Coelho', 'Dan Brown', 'Margaret Mitchell', 'Jane Austen']
}

ratings = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'book_id': [1, 2, 2, 3, 1, 3, 2, 4, 1, 5],
    'rating': [5, 4, 5, 3, 4, 4, 5, 3, 2, 4]
}

recommender_system = BookRecommenderSystem(books, ratings)

book_recommender_gui = BookRecommenderGUI(recommender_system)