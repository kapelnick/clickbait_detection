# -*- coding: utf-8 -*-

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the trained classifier from the file
model_filename = 'trained_model.pkl'
loaded_classifier = joblib.load(model_filename)

# Load BERT model and tokenizer
model_name = 'nlpaueb/bert-base-greek-uncased-v1'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def has_curiosity_gap(article_text):#ερωτηματικές αντωνυμίες
    curiosity_keywords = ['...', 'Ανακαλύψτε', 'ΑΝΑΚΑΛΥΨΤΕ', 'Μάθε', 'ΜΑΘΕ', 'Μυστικά', 'ΜΥΣΤΙΚΑ', 'Τρόποι', 'ΤΡΟΠΟΙ', 'ΠΩΣ', 'Πώς', 'Πως', 'ΠΟΤΕ', 'Πότε', 'Ποια', 'ΠΟΙΑ', 'Γιατί', 'ΓΙΑΤΙ', 'Κάποιο', 'ΚΑΠΟΙΟ', 'Κάποιος', 'ΚΑΠΟΙΟΣ', 'Κάποια', 'ΚΑΠΟΙΑ', 'Ποιος', 'ΠΟΙΟΣ', 'ΤΙ', 'Τι', 'ΒΡΕΣ', 'Βρες', 'ΔΙΑΛΕΞΕ', 'Διάλεξε', 'Πόση', 'ΠΟΣΗ', 'Ποιο', 'ΠΟΙΟ', 'Πόσο', 'ΠΟΣΟ', 'Ποιον', 'ΠΟΙΟΝ', 'Δες', 'ΔΕΣ']
    
    return any(keyword in article_text for keyword in curiosity_keywords)

def is_numbered_list(title):
    # Check if the title is not empty or contains only whitespace
    if title and title.strip():
        return title.strip().split()[0].isdigit()
    return False  # If title is empty or contains only whitespace, not a numbered list


def get_sentiment_score(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.last_hidden_state.mean(dim=1).squeeze()
        softmax = torch.nn.functional.softmax(logits, dim=0)
    return softmax[1].item() - softmax[0].item()

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        try:
            title = lines[0].strip()
        except IndexError:
            title = 'None'

        try:
            date_time = lines[4].strip().split(" ")
            date = date_time[0]
            time = date_time[1]
        except IndexError:
            date = None
            time = None

        try:
            article_text = ' '.join(lines[6:]).strip()
        except IndexError:
            article_text = ''

        return {
            'Title': title,
            'Title length': len(title),
            'Date': date,
            'Time': time,
            'Article_Text': article_text,
            'Article_Text_Length': len(article_text)
        }

def create_dataframe(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            article_data = process_text_file(file_path)
            # Exclude articles with length < 600
            if len(article_data['Article_Text']) >= 600:
                data.append(article_data)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    directory_path = r"C:\Users\kapel\Desktop\click_bait_data\new_url"
    df = create_dataframe(directory_path)
    
    # Calculate sentiment for the "Title" column
    df.insert(2, 'Title Sentiment', df['Title'].apply(lambda x: get_sentiment_score(x, model, tokenizer)))
    #Calculate sentiment for the "Article_Text" column
    df.insert(7, 'Article Text Sentiment', df['Article_Text'].apply(lambda x: get_sentiment_score(x, model, tokenizer)))
    
    # Calculate the cosine similarity between the "Title Sentiment" and "Article Text Sentiment"
    df['Title-Text Sentiment Similarity'] = df.apply(lambda row: cosine_similarity(
        [[row['Title Sentiment']]],
        [[row['Article Text Sentiment']]]
    )[0][0], axis=1)
    
    # Calculate embeddings for the title and the text
    df['Title Embedding'] = df['Title'].apply(lambda x: model(**tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=512))['last_hidden_state'].mean(dim=1).squeeze().detach().numpy())
    df['Text Embedding'] = df['Article_Text'].apply(lambda x: model(**tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=512))['last_hidden_state'].mean(dim=1).squeeze().detach().numpy())

    # Calculate the cosine similarity between the title and the text embeddings
    df.insert(9, 'Title-Text Similarity', df.apply(lambda row: cosine_similarity([row['Title Embedding']], [row['Text Embedding']])[0][0], axis=1))
    
    # Drop the "Title Embedding" and "Text Embedding" columns if needed
    df.drop(columns=['Title Embedding', 'Text Embedding'], inplace=True, errors='ignore')
    
    # Check for Curiosity Gap
    df['Curiosity Gap'] = df['Title'].apply(has_curiosity_gap)

    # Check for Numbered Lists
    df['Numbered List'] = df['Title'].apply(is_numbered_list)

    # Save the DataFrame to an Excel file
    output_excel_file = "output_dataframe_newURL.xlsx"
    df.to_excel(output_excel_file, index=False, encoding='utf-8')

    print(f"Data Frame saved to {output_excel_file}.")
    

new_df = pd.read_excel("output_dataframe_newURL.xlsx")

# Separate features (X_new) from the new DataFrame
X_new = new_df[['Title length', 'Title Sentiment', 'Article_Text_Length', 'Article Text Sentiment', 'Title-Text Sentiment Similarity', 'Title-Text Similarity', 'Curiosity Gap', 'Numbered List']]

# Assuming you have the new features (X_new) for prediction
# Make predictions using the loaded classifier
predicted_click_bait = loaded_classifier.predict(X_new)

# Insert the "Click-Bait" column
new_df['Predicted Clickbait Score'] = loaded_classifier.predict(X_new)

new_df.to_excel(output_excel_file, index=False, encoding='utf-8')
print(f"New Data Frame saved to {output_excel_file}.\n")

# The predicted_click_bait variable now contains the predicted label for the new URL:
if predicted_click_bait == 1:
    print('\n' + '\n' + '\n' + '\n' + '\n')
    print("Predicted Clickbait score is 1. The article is classified as click-bait.")
else:
    print('\n' + '\n' + '\n' + '\n' + '\n')
    print("Predicted Clickbait score is 0. The article is not classified as click-bait.")
