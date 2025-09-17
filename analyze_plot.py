import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sentiment_results.csv")
counts = df['sentiment'].value_counts()

counts.plot(kind='bar', title='Customer Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.savefig('sentiment_bar.png')
plt.show()
