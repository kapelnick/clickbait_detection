# -*- coding: utf-8 -*-
import re
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import warnings
warnings.filterwarnings("ignore")

# Load BERT model and tokenizer
model_name = 'nlpaueb/bert-base-greek-uncased-v1'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the English language model
nlp = spacy.load("el_core_news_sm")

def contains_adjectives(title):
    # Process the title using spaCy
    doc = nlp(title)
    
    # Check if there are any adjectives in the title
    for token in doc:
        if token.pos_ == "ADJ":
            return 'TRUE'
    # If no adjectives are found, return 'no'
    return 'FALSE'

def normalize_sentiment(score):
    return round((score + 1) / 2, 2)

def has_curiosity_gap(article_text): #ερωτηματικές αντωνυμίες κλπ
    curiosity_keywords = ['Ανακαλύψτε', 'ΑΝΑΚΑΛΥΨΤΕ', 'Μάθε', 'ΜΑΘΕ', 'Μυστικά', 'ΜΥΣΤΙΚΑ', 'Τρόποι', 'ΤΡΟΠΟΙ', 'ΠΩΣ', 'Πώς', 'Πως', 'ΠΟΤΕ', 'Πότε', 'Ποια', 'ΠΟΙΑ', 'Γιατί', 'ΓΙΑΤΙ', 'Κάποιο', 'ΚΑΠΟΙΟ', 'Κάποιος', 'ΚΑΠΟΙΟΣ', 'Κάποια', 'ΚΑΠΟΙΑ', 'Ποιος', 'ΠΟΙΟΣ', 'ΤΙ', 'Τι', 'ΒΡΕΣ', 'Βρες', 'ΔΙΑΛΕΞΕ', 'Διάλεξε', 'Πόση', 'ΠΟΣΗ', 'Ποιο', 'ΠΟΙΟ', 'Πόσο', 'ΠΟΣΟ', 'Ποιον', 'ΠΟΙΟΝ', 'Δες', 'ΔΕΣ' 'ΠΟΥ', 'πού']
    
    return any(keyword in article_text for keyword in curiosity_keywords)

def has_punctuation(title):
    punctuation_marks = [';', '!', '...']  # List of punctuation marks to check for
    
    # Check if any of the punctuation marks are present in the title
    if any(mark in title for mark in punctuation_marks):
        return True
    else:
        return False

def starts_with_number(title):
    # Check if the title is not empty and the first character is a digit
    if title and title[0].isdigit():
        return True
    else:
        return False


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
    directory_path = r"C:\Users\kapel\Desktop\Git\clickbait_detection\clickbait_detection\naft"
    df = create_dataframe(directory_path)

    decimal_places = 2  # Number of decimal places to round to
    
    # Calculate sentiment for the "Title" column
    df.insert(2, 'Title Sentiment', df['Title'].apply(lambda x: get_sentiment_score(x, model, tokenizer)))
    df['Title Sentiment'] = df['Title Sentiment'].apply(normalize_sentiment)
    
    #Calculate sentiment for the "Article_Text" column
    df.insert(7, 'Article Text Sentiment', df['Article_Text'].apply(lambda x: get_sentiment_score(x, model, tokenizer)))
    df['Article Text Sentiment'] = df['Article Text Sentiment'].apply(normalize_sentiment)
    
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
    df['Numbered List'] = df['Title'].apply(starts_with_number)
    
    # Check for Numbered Lists
    df['Adjectives'] = df['Title'].apply(contains_adjectives)

    # # Check for Excessive Punctuation
    df['Punctuation'] = df['Title'].apply(has_punctuation)
    
    # Insert the "Click-Bait" column based on conditions
    df['Clickbait Score'] = df.apply(lambda row: 1 if row['Curiosity Gap'] or row['Numbered List'] else 0, axis=1)
    
    # Save the DataFrame to an Excel file
    output_excel_file = "output_dataframe2.xlsx"
    df.to_excel(output_excel_file, index=False, encoding='utf-8')

    print(f"DataFrame saved to {output_excel_file}.")