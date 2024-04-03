import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk

# Load data
faqs_df = pd.read_csv('faqs.csv', encoding='cp1252')
info_df = pd.read_csv('info.csv', encoding='cp1252')
menu_df = pd.read_csv('menu.csv', encoding='cp1252')

# NLTK setup for text processing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and not word.lower() in stop_words]
    return ' '.join(lemmatized_tokens)

def get_general_info(keyword):
    if keyword in ['hours', 'contact', 'location', 'website']:
        response = {
            'hours': "We're open from 5 PM to 11 PM on weekdays, and 12 PM to 1 AM on weekends.",
            'contact': f"You can contact us at: {info_df.iloc[0]['Phone Number']}",
            'location': f"Our address is: {info_df.iloc[0]['Address']}",
            'website': f"Visit our website for more info: {info_df.iloc[0]['Website']}"
        }
        return response[keyword]
    return "For more details, please visit our website or contact us directly."

def get_menu_item_recommendation(category=None):
    if category:
        filtered_menu = menu_df[menu_df['Category'].str.contains(category, case=False, na=False)]
        if not filtered_menu.empty:
            recommendation = filtered_menu.sample()
        else:
            return "Sorry, I couldn't find any items in that category."
    else:
        recommendation = menu_df.sample()
    item = recommendation.iloc[0]
    item_description = item['Description'] if pd.notnull(item['Description']) else "a delicious choice"
    return f"How about trying our {item['Item Name']}? It's {item_description} at ${item['Price']}."

def get_closest_faq(user_input):
    user_input_preprocessed = preprocess_text(user_input)
    faqs_preprocessed = [preprocess_text(q) for q in faqs_df['Question']]
    vectorizer = TfidfVectorizer().fit(faqs_preprocessed)
    faqs_vectorized = vectorizer.transform(faqs_preprocessed)
    user_input_vectorized = vectorizer.transform([user_input_preprocessed])
    similarities = cosine_similarity(user_input_vectorized, faqs_vectorized).flatten()
    closest_idx = np.argmax(similarities)
    if similarities[closest_idx] > 0.1:
        return faqs_df.iloc[closest_idx]['Answer']
    else:
        return "I'm sorry, I couldn't find an answer to your question. Can you ask in a different way?"

def get_response(user_input):
    print(f"You asked: \"{user_input}\"")  # Echo the user's query
    user_input_lower = user_input.lower()

    if 'hours' in user_input_lower or 'open' in user_input_lower:
        return get_general_info('hours')
    elif 'contact' in user_input_lower or 'phone' in user_input_lower or 'email' in user_input_lower:
        return get_general_info('contact')
    elif 'location' in user_input_lower or 'address' in user_input_lower:
        return get_general_info('location')
    elif 'website' in user_input_lower:
        return get_general_info('website')
    elif 'cocktail' in user_input_lower or 'drink' in user_input_lower:
        return get_menu_item_recommendation(category='Cocktails')
    elif 'menu' in user_input_lower or 'eat' in user_input_lower or 'food' in user_input_lower:
        return get_menu_item_recommendation()
    elif 'outdoor seating' in user_input_lower:
        return "We do not offer outdoor seating."
    else:
        return get_closest_faq(user_input)

def create_gui():
    window = tk.Tk()
    window.title("Sweet Georgia's Juke Joint Chatbot")

    # Chat history display
    chat_history_text = tk.Text(window, height=15, font=("Trebuchet MS", 12))  # Set font family and size for chat history
    chat_history_text.pack(padx=10, pady=10)

    # Define tags for styling
    chat_history_text.tag_config("you", foreground="darkorange", font=("Trebuchet MS", 12, "bold"))
    chat_history_text.tag_config("chatbot", foreground="black", font=("Trebuchet MS", 12, "bold"))

    # Text entry box
    user_input_entry = tk.Entry(window, font=("Trebuchet MS", 12), width=50)
    user_input_entry.pack(padx=10, pady=10)

    # Send button
    send_button = tk.Button(window, text="Send", font=("Trebuchet MS", 12), bg="darkorange", relief="raised", command=lambda: send_message())
    send_button.pack(padx=10, pady=10)
    
    window.configure(bg="antiquewhite")  # Set background color for the main window
    chat_history_text.configure(bg="oldlace")
    
    # Welcome message
    chat_history_text.insert(tk.END, "Welcome to Sweet Georgia's Juke Joint Chatbot!\nHow can I assist you today? (Type 'quit' to exit)\n")

    def send_message():
        user_input = user_input_entry.get()
        chat_history_text.insert(tk.END, "You: ", "you")  # Apply "you" tag to the "You:" label
        chat_history_text.insert(tk.END, user_input + "\n")
        response = get_response(user_input)
        chat_history_text.insert(tk.END, "Chatbot: ", "chatbot")  # Apply "chatbot" tag to the "Chatbot:" label
        chat_history_text.insert(tk.END, response + "\n")
        user_input_entry.delete(0, tk.END)
        if user_input.lower() == 'quit':
            window.destroy()  # Close the window when user types "quit"
            return
        
    window.mainloop()


if __name__ == "__main__":
    create_gui()