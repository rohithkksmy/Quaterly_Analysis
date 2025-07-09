import streamlit as st
from deeppavlov import build_model, configs

# Load the sentiment analysis model
sentiment_model = build_model(configs.classifiers.rubert_sentiment, download=True)

# Load the conversational model
dialogue_model = build_model(configs.dialogue.torch_dialogue, download=True)

# Streamlit UI setup
st.title("Real-Time Conversational Chatbot with Sentiment Analysis")
st.write("Type your message below and click 'Submit' to analyze sentiment!")

# Initialize chat history session state if not already initialized
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = []

# Display previous chat messages
for message in st.session_state.messages:
    if message['sender'] == 'user':
        st.write(f"**You**: {message['text']}")
    else:
        st.write(f"**Bot**: {message['text']}")

# Text input for user
user_input = st.text_input("You:", "")

# Add a Submit button to trigger sentiment analysis
submit_button = st.button("Submit")

# When user clicks submit, process the input
if submit_button and user_input:
    # Add user input to the chat history
    st.session_state.messages.append({'sender': 'user', 'text': user_input})
    
    # Analyze sentiment of the user's message
    sentiment = sentiment_model([user_input])  # Returns sentiment score
    sentiment_score = sentiment[0][0]  # 1 for positive, -1 for negative
    
    # Add sentiment score to sentiment_scores
    st.session_state.sentiment_scores.append(sentiment_score)

    # Display sentiment score
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative"
    st.write(f"Sentiment of your message: **{sentiment_label}** (Score: {sentiment_score})")

    # Get bot response using the conversational model
    bot_reply = dialogue_model([user_input])[0]  # Get bot response

    # Add bot reply to chat history
    st.session_state.messages.append({'sender': 'bot', 'text': bot_reply})

    # Display bot's response
    st.write(f"**Bot**: {bot_reply}")

# Display sentiment summary when Submit is clicked
if st.session_state.sentiment_scores:
    positive_count = sum(1 for score in st.session_state.sentiment_scores if score > 0)
    negative_count = sum(1 for score in st.session_state.sentiment_scores if score < 0)
    st.write(f"**Sentiment Summary**:")
    st.write(f"  - Positive Messages: {positive_count}")
    st.write(f"  - Negative Messages: {negative_count}")
