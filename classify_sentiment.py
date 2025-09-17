import pandas as pd
from ollama import chat

# Load sample comments
df = pd.read_csv("comments.csv")

results = []
for comment in df['comment']:
    prompt = f"Classify this customer comment as Positive, Negative, or Neutral:\n\n{comment}"
    resp = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    sentiment = resp['message']['content'].strip()
    results.append({"comment": comment, "sentiment": sentiment})

# Save classification
pd.DataFrame(results).to_csv("sentiment_results.csv", index=False)
print("Saved sentiment_results.csv")
