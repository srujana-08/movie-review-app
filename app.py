import tkinter as tk
from tkinter import messagebox
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Function to predict review sentiment
def predict_sentiment():
    review = entry.get()  # Get review from Entry widget
    if review:
        # Transform the review text using the scaler
        review_scaler = scaler.transform([review]).toarray()
        try:
            result = model.predict(review_scaler)
            # Display the result in a message box
            if result[0] == 0:
                messagebox.showinfo('Result', 'Negative Review')
            else:
                messagebox.showinfo('Result', 'Positive Review')
        except Exception as e:
            messagebox.showerror('Error', f'Prediction failed: {e}')
    else:
        messagebox.showwarning('Input Error', 'Please enter a movie review.')

# Create the Tkinter window
root = tk.Tk()
root.title("Movie Review Sentiment Analysis")

# Add a label
label = tk.Label(root, text="Enter Movie Review:")
label.pack(pady=10)

# Add an Entry widget for review input
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Add a Button widget to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_sentiment)
predict_button.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()
