from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔥 OpenAI setup
client = OpenAI(api_key="YOUR_API_KEY_HERE")

app = Flask(__name__)

# 🔥 Load FAQ
with open("faq.json", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# 🔥 NLP vectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(questions)

# 🔥 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.strip()

def chatbot(user_input):
    user_input_clean = clean_text(user_input)

    # 🔥 Greeting
    if user_input_clean in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today?"

    # 🔥 Keyword quick reply
    if "order" in user_input_clean:
        return "You can track your order in the Orders section."
    if "return" in user_input_clean:
        return "Go to your orders and click return option."
    if "refund" in user_input_clean:
        return "Refund will be processed within 5-7 business days."
    if "delivery" in user_input_clean:
        return "Delivery usually takes 3 to 5 days."

    # 🔥 NLP matching
    user_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vec, X)

    if similarity.max() >= 0.2:
        index = similarity.argmax()

        if isinstance(answers[index], list):
            return random.choice(answers[index])

        return answers[index]

    # 🔥 AI fallback (🔥 MAIN UPGRADE)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful e-commerce customer support assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return "Sorry, I'm unable to process your request right now."

# 🔥 Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    user_msg = request.form.get("msg", "")
    return jsonify({"reply": chatbot(user_msg)})

# 🔥 Run app
if __name__ == "__main__":
    app.run(debug=True)