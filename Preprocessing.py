import pandas as pd
from transformers import pipeline

# Load the AI models
tox_p = pipeline("text-classification", model="unitary/toxic-bert")
sent_p = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

# 1. Added r'' to handle Windows path (Option 1 from earlier)
df = pd.read_csv(r'C:\Users\shaun\Downloads\youtoxic_english_1000.csv')

def get_scores(text):
    # Ensure text is a string and handle potential empty values
    text = str(text) if pd.notnull(text) else ""
    
    # 2. Use truncation=True and max_length=128
    # This ensures the tokenizer cuts the text at the model's exact limit
    t_res = tox_p(text, truncation=True, max_length=512)[0]
    t_val = t_res['score'] if t_res['label'] == 'toxic' else 1 - t_res['score']
    
    # BERTweet specifically has a max_length of 128
    s_res = sent_p(text, truncation=True, max_length=128)[0]
    s_val = 0.8 if s_res['label'] == "NEG" else 0.2 if s_res['label'] == "POS" else 0.5
    
    return pd.Series([t_val, s_val])

print("Processing comments... this may take a few minutes.")
# Applying the fix
df[['tox_score', 'sent_score']] = df['Text'].apply(get_scores)

df.to_csv('prepared_data.csv', index=False)
print("Output generated: prepared_data.csv")