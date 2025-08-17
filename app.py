# ==============================================================================
# FLASK WEB APPLICATION FOR SMS SPAM DETECTION
# ==============================================================================
# To run this app:
# 1. Make sure you have Flask installed (`pip install Flask xgboost scikit-learn`)
# 2. Save this code as `app.py`
# 3. Ensure 'tfidf_vectorizer.joblib' and 'spam_classifier_model.joblib' are in the same directory.
# 4. Open your terminal in this directory and run: `python app.py`
# 5. Open your web browser and go to http://127.0.0.1:5000
# ==============================================================================
import joblib  # type: ignore
import string
import nltk  # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from flask import Flask, request, render_template_string


# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Text Preprocessing Function (Must be identical to the one used for training) ---
# We include this directly in the app script for simplicity.
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses a single text message."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Load Saved Model and Vectorizer ---
# Use a try-except block to handle potential errors if files are missing.
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('spam_classifier_model.joblib')
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    # Define dummy objects if loading fails, so the app can still start and show an error.
    tfidf_vectorizer, model = None, None

# --- HTML Template with Tailwind CSS for Styling ---
# We embed the HTML in the Python file for a single-file deployment.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Spam Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="w-full max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8 md:p-12">
        <div class="text-center">
            <h1 class="text-3xl md:text-4xl font-bold text-gray-800">📱 SMS Spam Detector</h1>
            <p class="text-gray-600 mt-3">Enter a message to see if our model thinks it's spam or ham.</p>
        </div>
        
        <form action="/" method="post" class="mt-8">
            <div class="mb-4">
                <textarea name="message" rows="6" class="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition" placeholder="Type or paste your message here...">{{ message_text }}</textarea>
            </div>
            <div class="text-center">
                <button type="submit" class="w-full md:w-auto bg-blue-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition transform hover:scale-105">
                    Check Message
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="mt-8 p-5 rounded-lg text-center 
                    {% if prediction == 'SPAM' %} bg-red-100 border border-red-300 text-red-800 
                    {% else %} bg-green-100 border border-green-300 text-green-800 {% endif %}">
            <h2 class="text-2xl font-bold">
                {% if prediction == 'SPAM' %}
                    🚨 This looks like SPAM!
                {% else %}
                    ✅ This looks like HAM.
                {% endif %}
            </h2>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="mt-8 p-5 rounded-lg text-center bg-yellow-100 border border-yellow-300 text-yellow-800">
            <p class="font-semibold">{{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --- Define App Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    message_text = ""
    error_message = None

    if model is None or tfidf_vectorizer is None:
        error_message = "Model or vectorizer failed to load. Please check the server logs."
        return render_template_string(HTML_TEMPLATE, error=error_message)

    if request.method == 'POST':
        message_text = request.form.get('message', '').strip()
        if message_text:
            # 1. Preprocess the input
            processed_input = preprocess_text(message_text)
            # 2. Vectorize the input
            vectorized_input = tfidf_vectorizer.transform([processed_input])
            # 3. Predict
            prediction_code = model.predict(vectorized_input)[0]
            prediction_result = 'SPAM' if prediction_code == 1 else 'HAM'
        else:
            error_message = "Please enter a message to check."

    return render_template_string(
        HTML_TEMPLATE, 
        prediction=prediction_result, 
        message_text=message_text,
        error=error_message
    )

# --- Run the App ---
if __name__ == '__main__':
    # The app will be accessible at http://127.0.0.1:5000
    app.run(debug=True)
