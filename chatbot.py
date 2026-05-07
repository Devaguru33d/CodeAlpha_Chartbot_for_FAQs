import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("faq.json") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chatbot(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return answers[index]

print("🤖 Chatbot started (type 'exit' to stop)\n")

while True:
    user = input("You: ")
    if user.lower() == "exit":
        print("Bot: Bye!")
        break
    print("Bot:", chatbot(user))