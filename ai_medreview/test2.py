import os
import math
import warnings
import numpy as np
import pandas as pd
import requests
from colorama import Fore, Style, init
from loguru import logger
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (
    lower_text, normalize_whitespace, remove_eol_characters,
    remove_punct, remove_stopwords, replace_phone_numbers
)
from nlpretext.social.preprocess import (
    remove_emoji, remove_hashtag, remove_mentions
)
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ai_medreview.sheethelper import SheetHelper
from ai_medreview.automation.git_merge import do_git_merge
from ai_medreview.params import DATA_PATH
from ai_medreview.utils import time_it

# =============================================================================
# ENV & CONFIGURATION
# =============================================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")
logger.add("/tmp/ai_medreview_debug.log", rotation="5000 KB")

# =============================================================================
# MODEL LOADING (Singleton Pattern)
# =============================================================================
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
QA_MODEL = "deepset/roberta-base-squad2"
EMOJI_MODEL = "SamLowe/roberta-base-go_emotions"
ZS_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# Load once globally. Use device=0 for GPU, device=-1 for CPU
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=-1)
ner_pipeline = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple", device=-1)
qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=-1)
emotion_pipeline = pipeline("text-classification", model=EMOJI_MODEL, top_k=1, device=-1)
zs_pipeline = pipeline("zero-shot-classification", model=ZS_MODEL, device=-1)

CATEGORIES = [
     "Appointment Booking and Online Systems",
     "Appointment Availability and Waiting Times",
     "Difficulty Getting Through on Phone",
     "Reception Staff Rude or Unhelpful",
     "Reception Staff Friendly and Helpful",
     "Prescriptions and Repeat Medication Issues",
     "Blood Tests and Results Delays",
     "Waiting Time in Surgery / Waiting Room",
     "Excellent Clinical Care and Thorough Explanation",
     "Rushed Consultation or Not Listened To",
     "Staff Kindness, Empathy and Compassion",
     "Staff Professionalism and Knowledge",
     "Vaccinations and Immunisations",
     "Telehealth / Phone Consultations",
     "Treatment Quality and Effectiveness",
     "Follow-up and Continuity of Care",
     "Overall Excellent Service and Practice",
     "Irrelevant / Unclassifiable / Noise"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def safe_extract_labels(results):
    """Handles both old and new transformers pipeline output formats."""
    extracted = []
    for res in results:
        if isinstance(res, list):
            extracted.append((res[0]["label"], res[0].get("score", 0.0)))
        else:
            extracted.append((res["label"], res.get("score", 0.0)))
    return extracted

@time_it
def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1c-811fFJYT9ulCneTZ7Z8b4CK4feEDRheR0Zea5--d0/edit#gid=0",
        sheet_id=0,
     )
    df = sh.gsheet_to_df()
    df.columns = [
         "submission_id", "respondent-id", "time", "rating", "free_text",
         "do_better", "pcn", "surgery", "campaing_id", "logic",
         "campaign_rating", "campaign_freetext",
     ]
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df.sort_values(by="time", inplace=True)
    return df

@time_it
def word_count(df):
    df["free_text_len"] = df["free_text"].astype(str).str.split().str.len()
    df["do_better_len"] = df["do_better"].astype(str).str.split().str.len()
    return df

@time_it
def add_rating_score(data):
    rating_map = {"Very good": 5, "Good": 4, "Neither good nor poor": 3, "Poor": 2, "Very poor": 1}
    data["rating_score"] = data["rating"].map(rating_map)
    return data

# =============================================================================
# CORE NLP PIPELINES (Bug-Fixed & Length-Validated)
# =============================================================================
@time_it
def sentiment_analysis(data, column):
    texts = data[column].astype(str).replace(["", "nan"], np.nan).tolist()
    valid_mask = ~pd.isna(texts)
    valid_indices = np.where(valid_mask)[0]
    valid_texts = [texts[i] for i in valid_indices]

    n = len(data)
    final_labels = ["neutral"] * n
    final_scores = [0.0] * n

    if valid_texts:
        results = sentiment_pipeline(valid_texts, batch_size=32)
        for i, res in enumerate(results):
            lbl, sc = safe_extract_labels([res])[0]
            final_labels[valid_indices[i]] = lbl
            final_scores[valid_indices[i]] = sc

    data[f"sentiment_{column}"] = final_labels
    data[f"sentiment_score_{column}"] = final_scores
    return data

@time_it
def classification_pipeline(data, column, label_col_name, batch_size=16):
    texts = data[column].astype(str).replace(["", "nan"], np.nan).tolist()
    valid_mask = ~pd.isna(texts)
    valid_indices = np.where(valid_mask)[0]
    valid_texts = [texts[i] for i in valid_indices]

    n = len(data)
    labels = [""] * n
    if valid_texts:
        results = zs_pipeline(valid_texts, CATEGORIES, batch_size=batch_size)
        for i, res in enumerate(results):
            labels[valid_indices[i]] = res["labels"][0]
    data[label_col_name] = labels
    return data

@time_it
def emotion_classification(data, column):
    texts = data[column].astype(str).replace(["", "nan"], np.nan).tolist()
    valid_mask = ~pd.isna(texts)
    valid_indices = np.where(valid_mask)[0]
    valid_texts = [texts[i] for i in valid_indices]

    n = len(data)
    emotions = [np.nan] * n
    if valid_texts:
        results = emotion_pipeline(valid_texts, batch_size=16)
        for i, res in enumerate(results):
            label = res[0]["label"] if isinstance(res, list) else res["label"]
            emotions[valid_indices[i]] = label
    data[f"emotion_{column}"] = emotions
    return data

@time_it
def question_answering(data, column):
    texts = data[column].astype(str).replace(["", "nan"], np.nan).tolist()
    valid_mask = ~pd.isna(texts)
    valid_indices = np.where(valid_mask)[0]
    valid_texts = [texts[i] for i in valid_indices]

    n = len(data)
    qa_results = [np.nan] * n
    if valid_texts:
        questions = {
             "free_text": "Please tell us why you feel this way?",
             "do_better": "Is there anything that would have made your experience better?"
         }.get(column, "What is this about?")
        inputs = [{"question": questions, "context": t} for t in valid_texts]
        results = qa_pipeline(inputs, batch_size=8)
        for i, res in enumerate(results):
            qa_results[valid_indices[i]] = {"answer": res["answer"], "score": res.get("score", 0)}
    data[f"{column}_qa"] = qa_results
    return data

@time_it
def anonymize_names(texts):
    valid_mask = [isinstance(t, str) and t.strip() for t in texts]
    valid_indices = np.where(valid_mask)[0]
    valid_texts = [texts[i] for i in valid_indices]

    n = len(texts)
    anonymized = list(texts)
    if valid_texts:
        ner_results = ner_pipeline(valid_texts, batch_size=16)
        for i, entities in enumerate(ner_results):
            orig_text = valid_texts[i]
            anon_text = orig_text
            for entity in entities:
                if entity["entity_group"] == "PER":
                    anon_text = anon_text.replace(entity["word"], "[*PERSON*]")
            anonymized[valid_indices[i]] = anon_text
    return anonymized

@time_it
def get_person_names(texts):
    valid_mask = [isinstance(t, str) and t.strip() for t in texts]
    valid_indices = np.where(valid_mask)[0]
    valid_texts = [texts[i] for i in valid_indices]

    n = len(texts)
    names_list = [None] * n
    if valid_texts:
        ner_results = ner_pipeline(valid_texts, batch_size=16)
        for i, entities in enumerate(ner_results):
            names = [e["word"] for e in entities if e["entity_group"] == "PER"]
            names_list[valid_indices[i]] = names if names else None
    return names_list

# =============================================================================
# TEXT PROCESSING & UTILS
# =============================================================================
def text_preprocessing(text):
    if pd.isna(text):
        return np.nan
    pre = Preprocessor()
    pre.pipe(remove_mentions)
    pre.pipe(remove_hashtag)
    pre.pipe(remove_eol_characters)
    pre.pipe(remove_punct)
    pre.pipe(normalize_whitespace)
    pre.pipe(replace_phone_numbers, args={"country_to_detect": ["GB", "FR"], "replace_with": "[*PHONE*]"})
    return pre.run(text)

@time_it
def clean_data(df):
    cleaned_df = df.copy()
    cleaned_df.loc[cleaned_df["do_better_len"] < 6, "do_better"] = np.nan
    cleaned_df.loc[cleaned_df["free_text_len"] < 6, "free_text"] = np.nan
    cleaned_df.loc[cleaned_df["do_better_len"] < 6, "improvement_labels"] = np.nan
    cleaned_df.loc[cleaned_df["free_text_len"] < 6, "feedback_labels"] = np.nan
    return cleaned_df

@time_it
def concat_save_final_df(processed_df, new_df):
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    combined_data.sort_values(by="time", inplace=True, ascending=True)
    combined_data.to_csv(f"{DATA_PATH}/data_v2.csv", encoding="utf-8", index=False)
    logger.info(f"💾 data_v2.csv saved to: {DATA_PATH}")

@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data_v2.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=False)
    return df

def send_alert_webhook(number):
    webhook_url = "https://n8n-render-0yda.onrender.com/webhook/2d117d97-90cd-4c67-a7c3-b5da1d31a8f2"
    response = requests.post(webhook_url, json={"number": number}, timeout=10)
    if response.status_code == 200:
        logger.info(f"Webhook sent to n8n with ** {number} ** new responses processed.")
    else:
        logger.warning(f"Failed to send data: {response.status_code}, {response.text}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    logger.info("▶️ AI Medreview FFT - MAKE DATA - Started")

    raw_data = load_google_sheet()
    processed_data = load_local_data()

     # Only process new rows
    data = raw_data[~raw_data.index.isin(processed_data.index)]
    logger.info(f"🆕 New rows to process: {data.shape[0]}")

    if data.shape[0] == 0:
        logger.error("❌ Make Data terminated - No new rows")
        exit()

     # Initial processing
    data = word_count(data)
    data = add_rating_score(data)

     # Entity Recognition & Anonymization
    logger.info("🫥 Extracting & Anonymizing Names with Transformer")
    data["free_text_PER"] = get_person_names(data["free_text"].tolist())
    data["do_better_PER"] = get_person_names(data["do_better"].tolist())

    data["free_text"] = anonymize_names(data["free_text"].tolist())
    data["do_better"] = anonymize_names(data["do_better"].tolist())

     # Text Preprocessing
    logger.info("📗 Text Preprocessing with NLPretext")
    data["free_text"] = data["free_text"].apply(text_preprocessing)
    data["do_better"] = data["do_better"].apply(text_preprocessing)

     # Sentiment
    logger.info("💛 Sentiment Analysis")
    data = sentiment_analysis(data, "free_text")
    data = sentiment_analysis(data, "do_better")

     # Classification
    logger.info("🏷️ Feedback & Improvement Classification")
    data = classification_pipeline(data, "free_text", "feedback_labels", batch_size=16)
    data = classification_pipeline(data, "do_better", "improvement_labels", batch_size=16)

     # Emotion
    logger.info("🎭 Emotion Classification")
    data = emotion_classification(data, "free_text")
    data = emotion_classification(data, "do_better")

     # Question Answering
    logger.info("❓ Question Answering")
    data = question_answering(data, "free_text")
    data = question_answering(data, "do_better")

    logger.info("✅ Data pre-processing completed")
    concat_save_final_df(processed_data, data)

    do_git_merge()
    logger.info("👍 Pushed to GitHub - Master Branch")
    logger.info("🎉 Successful Run completed")
