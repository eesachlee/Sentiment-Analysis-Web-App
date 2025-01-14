import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('final_logistic_regression_model.pkl')
vectorizer = joblib.load('final_tfidf_vectorizer.pkl')

# Set the title and some styling for the app
st.title('Sentiment Analysis Web App')

# Add some description with custom styling
st.markdown("""
    <style>
        .title {
            font-size: 30px;
            font-weight: bold;
            color: #1E90FF;
        }
        .description {
            font-size: 18px;
            color: #555555;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
        }
    </style>
    <div class="title">Sentiment Analysis Web App</div>
    <div class="description">Enter any text, and I will predict the sentiment (Positive, Negative, or Neutral) of your text.</div>
""", unsafe_allow_html=True)

# Create a text input for the user
user_input = st.text_area("Enter text for sentiment analysis:")

# Function to predict sentiment
def predict_sentiment(text):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    # Make prediction
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Display prediction when the user submits text
if user_input:
    prediction = predict_sentiment(user_input)
    
    # Display the result with emojis
    if prediction == 'Positive':
        st.markdown("<div class='prediction' style='color: green;'>😊 Positive</div>", unsafe_allow_html=True)
    elif prediction == 'Negative':
        st.markdown("<div class='prediction' style='color: red;'>😞 Negative</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction' style='color: gray;'>😐 Neutral</div>", unsafe_allow_html=True)

    # Additional feature: Provide a feedback option
    st.write("How accurate do you think the prediction is?")
    feedback = st.radio("Rate the prediction:", ("Excellent", "Good", "Fair", "Poor"))
    st.write(f"Thank you for your feedback: {feedback}")

# Display an informative message
st.markdown("""
    <style>
        .footer {
            font-size: 14px;
            color: #999999;
            margin-top: 50px;
        }
    </style>
    <div class="footer">Made with ❤️ by Your Name</div>
""", unsafe_allow_html=True)
