import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import process
import os

# ============================
# ⚙️ Configs
# ============================
MATCH_THRESHOLD = 75 # Match score threshold (0-100)

# ============================
# 🧠 Load & Train Chatbot Model
# ============================
if os.path.exists("English_Chatbot.csv"):
    try:
        chat_df = pd.read_csv("English_Chatbot.csv", on_bad_lines='skip')
        chat_X = chat_df['input']
        chat_y = chat_df['chatbot']

        chat_vectorizer = TfidfVectorizer()
        chat_X_vec = chat_vectorizer.fit_transform(chat_X)

        chat_model = LogisticRegression()
        chat_model.fit(chat_X_vec, chat_y)

        chat_dict = dict(zip(chat_df['input'].str.lower(), chat_df['chatbot']))
    except Exception as e:
        st.error(f"❌ Error loading 'Ananth.csv': {e}")
        st.stop()
else:
    st.error("❌ 'Ananth.csv' file not found.")
    st.stop()

# ===============================
# 🧠 Load & Train Sentiment Model
# ===============================
if os.path.exists("friendly_emotion_chatbot.csv"):
    try:
        emotion_df = pd.read_csv("friendly_emotion_chatbot.csv", on_bad_lines='skip')
        emo_X = emotion_df['input']
        emo_y = emotion_df['emotion']

        emo_vectorizer = TfidfVectorizer()
        emo_X_vec = emo_vectorizer.fit_transform(emo_X)

        emo_model = LogisticRegression()
        emo_model.fit(emo_X_vec, emo_y)

        emo_dict = dict(zip(emotion_df['input'].str.lower(), emotion_df['emotion']))
    except Exception as e:
        st.error(f"❌ Error loading 'friendly_emotion_chatbot.csv': {e}")
        st.stop()
else:
    st.error("❌ 'friendly_emotion_chatbot.csv' file not found.")
    st.stop()

# ========================
# 🤖 Prediction Functions
# ========================
def get_chat_response(user_input):
    user_vec = chat_vectorizer.transform([user_input])
    pred = chat_model.predict(user_vec)[0]
    match = process.extractOne(user_input.lower(), chat_dict.keys())
    if match and match[1] >= MATCH_THRESHOLD:
        return chat_dict[match[0]]
    else:
        return pred

def get_emotion(user_input):
    user_vec = emo_vectorizer.transform([user_input])
    pred = emo_model.predict(user_vec)[0]
    match = process.extractOne(user_input.lower(), emo_dict.keys())
    if match and match[1] >= MATCH_THRESHOLD:
        return emo_dict[match[0]]
    else:
        return pred

# ===================
# 🎨 Page UI Layout
# ===================
st.set_page_config("Chatbot", layout="wide")

st.markdown("""
    <style>
    .main-container {
        background: linear-gradient(to bottom right, #e3f2fd, #f3e5f5);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .question-box {
        background-color:#1976D2;
        color:#FFFFFF;
        padding:10px;
        border-radius:10px;
        margin:5px 0;
        font-size:14px;
    }
    .response-box {
        background-color:#2E7D32;
        color:#FFFFFF;
        padding:12px;
        border-radius:15px;
        margin-top:15px;
        font-size:16px;
    }
    .emotion-box {
        color:#FFFFFF;
        padding:12px;
        border-left:5px solid #fff;
        border-radius:8px;
        font-size:16px;
        margin-top:15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='text-align: center; color: #3F51B5;'>🤖 Friendly Fun Chatbot & Sentiment Detector</h2>
<p style='text-align: center; font-size:18px;'>Talk like a friend. I reply & feel your emotion too 💬❤️</p>
""", unsafe_allow_html=True)

# ======================
# 🧠 Session State Init
# ======================
if "history" not in st.session_state:
    st.session_state.history = []
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []

# ====================
# 🎨 Emotion Colors
# ====================
emotion_color_map = {
    "happy": "#4CAF50",
    "sad": "#E53935",
    "stress": "#FF9800",
    "emotional": "#9C27B0",
    "angry": "#F44336",
    "love": "#EC407A",
    "depression": "#455A64"
}

# 🎧 Spotify playlist embed links
spotify_embed_links = {
    "happy": "https://open.spotify.com/embed/playlist/2P4Wmt03IQs4DTXVvncReg",
    "sad": "https://open.spotify.com/embed/playlist/0AyOLKzLZZmlliok7bu1mp",
    "stress": "https://open.spotify.com/embed/playlist/1YQBOoZJHFJzHx2Pm6sd4w",
    "depression": "https://open.spotify.com/embed/playlist/4GX1yWidUcdCCuIJZSX4Rc",
    "love": "https://open.spotify.com/embed/playlist/6qEZQ1OXaYJaCJHeSHVxO7",
    "angry": "https://open.spotify.com/embed/playlist/2Lnt48sabnkqVZqyFvpbq9",
    "emotional": "https://open.spotify.com/embed/track/2Osew72vzCf361dtOEF7bB"
}

# =====================
# 📱 Layout - Chat UI
# =====================
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<h4>📜 Your Asked Questions</h4>", unsafe_allow_html=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.user_questions = []
        st.rerun()

    for question in st.session_state.user_questions:
        st.markdown(f"<div class='question-box'><b>You:</b> {question}</div>", unsafe_allow_html=True)

with col2:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your message:", placeholder="Yedhachum Pesu Daa")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        bot_reply = get_chat_response(user_input)
        emotion = get_emotion(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_reply))
        st.session_state.user_questions.append(user_input)

        st.markdown(f"""
            <div class='emotion-box' style="background-color:{emotion_color_map.get(emotion, '#616161')};">
            🔔 <b>Detected Emotion:</b> {emotion.upper()}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='response-box'>
                <b>Bot:</b> {bot_reply}
            </div>
        """, unsafe_allow_html=True)

        # 🎧 Embed Spotify player if emotion matched
        spotify_url = spotify_embed_links.get(emotion)
        if spotify_url:
            st.markdown(f"""
                <iframe style="border-radius:12px; margin-top:10px;" 
                        src="{spotify_url}" 
                        width="100%" 
                        height="152" 
                        frameborder="0" 
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                        loading="lazy">
                </iframe>
            """, unsafe_allow_html=True)

# =================
# 🔚 Footer
# =================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center><small style='color:#555;'>Made with ❤️ using Streamlit • Chat + Mood Aware</small></center>", unsafe_allow_html=True)
