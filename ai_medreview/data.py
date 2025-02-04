import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from colorama import Back, Fore, Style, init
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (
    lower_text,
    normalize_whitespace,
    remove_eol_characters,
    remove_punct,
    remove_stopwords,
    replace_phone_numbers,
)
from nlpretext.social.preprocess import remove_emoji, remove_hashtag, remove_mentions
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ai_medreview.sheethelper import SheetHelper

tqdm.pandas()

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

from ai_medreview.automation.git_merge import *
from ai_medreview.params import *
from ai_medreview.utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")

from loguru import logger

logger.add("log/debug.log", rotation="5000 KB")

# Select Classification Model - facebook/bart-large-mnli or FacebookAI/roberta-large-mnli

@time_it
def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1c-811fFJYT9ulCneTZ7Z8b4CK4feEDRheR0Zea5--d0/edit#gid=0",
        sheet_id=0,
    )
    data = sh.gsheet_to_df()
    data.columns = [
        "submission_id",
        "respondent-id",
        "time",
        "rating",
        "free_text",
        "do_better",
        "pcn",
        "surgery",
        "campaing_id",
        "logic",
        "campaign_rating",
        "campaign_freetext",
    ]

    data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S")

    data.sort_values(by="time", inplace=True)
    return data

@time_it
def word_count(df):
    df["free_text_len"] = df["free_text"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else np.nan
    )

    df["do_better_len"] = df["do_better"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else np.nan
    )

    return df


@time_it
def check_column_length(dataframe, column_name, word_count_length):
    # Iterate over each entry in the specified column
    for index, entry in enumerate(dataframe[column_name]):
        # Count the number of words in the entry
        word_count = len(str(entry).split())

        # Check if the word count is less than the specified limit
        if word_count < word_count_length:
            # Replace with NaN if the condition is met
            dataframe.at[index, column_name] = np.nan

    return dataframe


sentiment_task = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def sentiment_analysis(data, column):
    logger.info("💛 Sentiment Analysis - Functions started.")

    # Initialize lists to store labels and scores
    sentiment = []
    sentiment_score = []

    # Iterate over DataFrame rows and classify text
    for index, row in tqdm(
        data.iterrows(),
        total=data.shape[0],
        desc="Analyzing Sentiment",
        colour="#e8c44d",
    ):
        freetext = row[column]
        sentence = str(freetext)
        sentence = sentence[:513]
        if pd.isna(sentence) or sentence == "":
            sentiment.append("neutral")
            sentiment_score.append(0)
        else:
            model_output = sentiment_task(sentence)
            sentiment.append(model_output[0]["label"])
            sentiment_score.append(model_output[0]["score"])

    # Add labels and scores as new columns
    data[f"sentiment_{column}"] = sentiment
    data[f"sentiment_score_{column}"] = sentiment_score

    return data


def sentiment_arc_analysis(data, column, chunk_size=6):

    # Initialize lists to store labels and scores
    sentiment = []
    sentiment_score = []
    sentiment_arc = []

    # Iterate over DataFrame rows and classify text
    for index, row in tqdm(
        data.iterrows(),
        total=data.shape[0],
        desc="Analyzing Sentiment Arc",
        colour="#50b2d4",
    ):
        temp_sentiment_arc = []
        freetext = row[column]
        words = str(freetext)
        words = words.split()

        # Calculate the number of chunks
        num_chunks = math.ceil(len(words) / chunk_size)

        for i in range(num_chunks):
            # Get the current chunk of words
            chunk = words[i * chunk_size : (i + 1) * chunk_size]

            # Join the chunk into a single string
            chunk_text = " ".join(chunk)

            # Calculate the sentiment polarity of the chunk
            if pd.isna(chunk_text) or chunk_text == "":
                sentiment_arc.append(0)
            else:
                model_output = sentiment_task(chunk_text)
                temp_sentiment = model_output[0]["label"]
                temp_sentiment_score = model_output[0]["score"]
                if temp_sentiment == "negative":
                    chunk_sentiment = -abs(temp_sentiment_score)
                elif temp_sentiment == "neutral":
                    chunk_sentiment = 0
                elif temp_sentiment == "positive":
                    chunk_sentiment = temp_sentiment_score

                temp_sentiment_arc.append(chunk_sentiment)

        sentiment_arc.append(temp_sentiment_arc)

    # Add labels and scores as new columns
    data[f"sentiment_arc_{column}"] = sentiment_arc

    return data


def cleanup_neutral_sentiment(df, column):
    logger.info("🧻 Cleanup_neutral_sentiment - if free_text and do_better isna()")

    cleaned_df = df.copy()
    cleaned_df.loc[
        (df[column].isnull()) | (df[column] == ""),
        [f"sentiment_score_{column}", f"sentiment_{column}"],
    ] = [0, "neutral"]

    return cleaned_df


ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
)

qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
# Function to anonymize names in text
@time_it
def question_answering(data, column):
    logger.info("🔍 Question Answering - Functions started.")

    output_list = []

    for _, row in tqdm(data.iterrows(), '🅾️ Answering', total=data.shape[0]):
        text = row[column]

        if column == "free_text":
            input_dict = {
            "question": "Please tell us why you feel this way?",
            "context": text,
            }
        elif column == "do_better":
            input_dict = {
            "question": "Is there anything that would have made your experience better?",
            "context": text,
            }

        # Check if free_text is a valid string and not empty or np.nan
        if isinstance(text, str) and text.strip():  # Check if it's a non-empty string
            # Process the free_text with your model and append the result
            result = qa_pipe(input_dict)
            output_list.append(result)
        else:
            # Append np.nan if free_text is empty, np.nan, or not a string
            output_list.append(np.nan)


    # Add labels and scores as new columns
    data[f"{column}_qa"] = output_list
    return data

def anonymize_names_with_transformers(text):

    # Check if the text is empty or not a string
    if not text or not isinstance(text, str):
        return text  # Return the text as-is if it's invalid or empty

    # Initialize the anonymized text
    anonymized_text = text

    try:
        # Run the NER pipeline on the valid input text
        entities = ner_pipeline(text)

        # Iterate over detected entities
        for entity in entities:
            # Check if the entity is classified as a person
            if entity["entity_group"] == "PER":
                # Replace the detected name with a placeholder
                anonymized_text = anonymized_text.replace(entity["word"], "[*PERSON*]")
    except ValueError as e:
        # Log the error for debugging
        print(f"Error processing text: {text}")
        raise e

    return anonymized_text


# Zer0-shot classification - do_better column
def batch_generator(data, column_name, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[column_name][i : i + batch_size]
        # Logging the batch content; you can comment this out or remove it in production
        yield batch, i  # Yield the batch and the starting index


@time_it
def feedback_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            framework="pt",  # specify PyTorch
            num_workers=0,  # Disable multiprocessing
        )  # -1 for CPU
    except Exception as e:
        print(f"Error in initializing the classifier pipeline: {e}")
        return data  # Exit if the pipeline cannot be created

    # Define the categories for classification
    categories = [
        "Staff Professionalism",
        "Communication Effectiveness",
        "Appointment Availability",
        "Waiting Time",
        "Facility Cleanliness",
        "Patient Respect",
        "Treatment Quality",
        "Staff Empathy and Compassion",
        "Administrative Efficiency",
        "Reception Staff Interaction",
        "Environment and Ambiance",
        "Follow-up and Continuity of Care",
        "Accessibility and Convenience",
        "Patient Education and Information",
        "Feedback and Complaints Handling",
        "Test Results",
        "Surgery Website",
        "Telehealth",
        "Vaccinations",
        "Prescriptions and Medication Management",
        "Mental Health Support",
    ]  # Include all your categories here

    # Initialize the list to store labels
    feedback_labels = [""] * len(data)  # Pre-fill with empty strings

    # Process batches
    total_batches = (
        len(data) + batch_size - 1
    ) // batch_size  # Calculate total number of batches
    for batch, start_index in tqdm(
        batch_generator(data, "free_text", batch_size),
        total=total_batches,
        desc="Processing batches",
    ):
        # Validate and filter batch data
        valid_sentences = [
            (sentence.strip(), idx)
            for idx, sentence in enumerate(batch)
            if isinstance(sentence, str) and sentence.strip()
        ]
        if not valid_sentences:
            continue  # Skip if no valid sentences are present

        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        try:
            # Perform classification
            model_outputs = classifier(list(sentences), categories, device="cpu")
            # Assign the most relevant category label
            for output, idx in zip(model_outputs, valid_indices):
                feedback_labels[start_index + idx] = output["labels"][0]
        except Exception as e:
            print(f"Error during classification: {e}")
            # Optionally, handle specific actions for failed classification, such as logging or retrying

    # Assign the computed labels back to the data
    data["feedback_labels"] = feedback_labels
    return data


@time_it
def improvement_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification", model=model, tokenizer=tokenizer, device=-1
        )  # -1 denotes CPU
    except Exception as e:
        print(f"Error initializing the classifier pipeline: {e}")
        return data  # Exit if the pipeline cannot be created

    # Define the labels for improvement categories
    improvement_labels_list = [
        "Staff Professionalism",
        "Communication Effectiveness",
        "Appointment Availability",
        "Waiting Time",
        "Facility Cleanliness",
        "Patient Respect",
        "Treatment Quality",
        "Staff Empathy and Compassion",
        "Administrative Efficiency",
        "Reception Staff Interaction",
        "Environment and Ambiance",
        "Follow-up and Continuity of Care",
        "Accessibility and Convenience",
        "Patient Education and Information",
        "Feedback and Complaints Handling",
        "Test Results",
        "Surgery Website",
        "Telehealth",
        "Vaccinations",
        "Prescriptions and Medication Management",
        "Mental Health Support",
    ]  # Your improvement labels

    # Initialize the list to store improvement labels
    improvement_labels = [""] * len(data)  # Pre-fill with empty strings

    total_batches = (
        len(data) + batch_size - 1
    ) // batch_size  # Calculate total number of batches
    for batch, start_index in tqdm(
        batch_generator(data, "do_better", batch_size),
        total=total_batches,
        desc="Processing batches",
    ):
        # Validate and filter batch data
        valid_sentences = [
            (sentence.strip(), idx)
            for idx, sentence in enumerate(batch)
            if isinstance(sentence, str) and sentence.strip()
        ]
        if not valid_sentences:
            continue  # Skip if no valid sentences are present

        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        try:
            # Classify the valid sentences
            model_outputs = classifier(
                list(sentences), improvement_labels_list, device="cpu"
            )
            # Update labels based on classification output
            for output, idx in zip(model_outputs, valid_indices):
                improvement_labels[start_index + idx] = output["labels"][0]
        except Exception as e:
            print(f"Error during classification: {e}")
            # Handle errors appropriately, possibly by logging or taking specific actions

    # Assign the computed labels back to the data
    data["improvement_labels"] = improvement_labels
    return data


@time_it
def add_rating_score(data):
    # Mapping dictionary
    rating_map = {
        "Very good": 5,
        "Good": 4,
        "Neither good nor poor": 3,
        "Poor": 2,
        "Very poor": 1,
    }

    # Apply the mapping to the 'rating' column
    data["rating_score"] = data["rating"].map(rating_map)
    return data


@time_it
def clean_data(df):
    logger.info("🧽 Clean data - delete feedback / Improvement Suggestions < 6 words.")

    # Copy the DataFrame to avoid modifying the original data
    cleaned_df = df.copy()
    # Apply the conditions and update the DataFrame
    cleaned_df.loc[cleaned_df["do_better_len"] < 6, "do_better"] = np.nan
    cleaned_df.loc[cleaned_df["free_text_len"] < 6, "free_text"] = np.nan
    cleaned_df.loc[cleaned_df["do_better_len"] < 6, "improvement_labels"] = np.nan
    cleaned_df.loc[cleaned_df["free_text_len"] < 6, "feedback_labels"] = np.nan
    return cleaned_df


@time_it
def concat_save_final_df(processed_df, new_df):
    logger.info("💾 Concat Dataframes to data.parquet successfully")
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    combined_data.sort_values(by="time", inplace=True, ascending=True)
    # combined_data.to_parquet(f"{DATA_PATH}/data.parquet", index=False)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", encoding="utf-8", index=False)
    print(f"💾 data.csv saved to: {DATA_PATH}")


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=False)
    return df


def text_preprocessing(text):

    preprocessor = Preprocessor()
    # preprocessor.pipe(lower_text)
    preprocessor.pipe(remove_mentions)
    preprocessor.pipe(remove_hashtag)
    # preprocessor.pipe(remove_emoji)
    preprocessor.pipe(remove_eol_characters)
    # preprocessor.pipe(remove_stopwords, args={'lang': 'en'})
    preprocessor.pipe(remove_punct)
    preprocessor.pipe(normalize_whitespace)
    preprocessor.pipe(
        replace_phone_numbers,
        args={"country_to_detect": ["GB", "FR"], "replace_with": "[*PHONE*]"},
    )
    text = preprocessor.run(text)

    return text


def send_alert_webhook(number):
    # URL of the webhook to which the data will be sent
    webhook_url = "https://n8n-render-0yda.onrender.com/webhook/2d117d97-90cd-4c67-a7c3-b5da1d31a8f2"
    # Create a dictionary to hold the data
    data = {"number": number}

    # Use the requests library to send a POST request with JSON data
    response = requests.post(webhook_url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        logger.info(f"Webhook sent to n8n with ** {number} ** new responses processed.")
    else:
        logger.info(f"Failed to send data: {response.status_code}, {response.text}")


@time_it
def create_monthyear(df):
    df["time"] = pd.to_datetime(df["time"])
    df["month_year"] = df["time"].dt.to_period("M")
    return df


# --- Emotion Classification Pipeline ----------------------------------------------------------------------------------
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)


@time_it
def emotion_classification(df, column, classifier):
    emotion = []

    # Total number of rows
    total_rows = df.shape[0]

    for index, row in tqdm(
        df.iterrows(), total=total_rows, desc="Analyzing Emotion", colour="#4088a9"
    ):
        sentence = row[column]
        if sentence == "" or pd.isna(sentence):
            emotion.append(np.nan)
        else:
            sentence = str(sentence)
            model_output = classifier(sentence, truncation=True, max_length=512)[0]
            model_output = list(model_output)
            model_output = model_output[0]["label"]
            emotion.append(model_output)

    df[f"emotion_{column}"] = emotion

    return df


def get_person_names_with_transformers(text):
    # Check if the text is empty or not a string
    if not text or not isinstance(text, str):
        return None  # Return None if the input is invalid or empty

    # Initialize a list to store person names
    person_names = []

    try:
        # Run the NER pipeline on the valid input text
        entities = ner_pipeline(text)

        # Iterate over detected entities
        for entity in entities:
            # Check if the entity is classified as a person
            if entity.get("entity_group") == "PER":  # Ensure key exists
                # Add the detected name to the list of person names
                person_names.append(
                    entity.get("word", "")
                )  # Use get() to avoid key errors
    except ValueError as e:
        # Log the error for debugging
        print(f"Error processing text: {text}")
        raise e

    # Return None if no person names were found, otherwise return the list of names
    return person_names if person_names else None


if __name__ == "__main__":
    logger.info("▶️ AI Medreview FFT - MAKE DATA - Started")

    # Load new data from Google Sheet
    raw_data = load_google_sheet()
    logger.info("Google Sheet data loaded")

    # Load local data.csv to dataframe
    processed_data = load_local_data()
    logger.info("data.csv Loadded")

    # Return new data for processing
    data = raw_data[~raw_data.index.isin(processed_data.index)]
    logger.info(f"🆕 New rows to process: {data.shape[0]}")

    if data.shape[0] != 0:

        data = word_count(data)  # word count
        data = add_rating_score(data)

        data["free_text_PER"] = data["free_text"].progress_apply(
            get_person_names_with_transformers
        )
        data["do_better_PER"] = data["do_better"].progress_apply(
            get_person_names_with_transformers
        )

        data = clean_data(data)

        logger.info("🫥 Annonymize with Transformer - free_text")
        data["free_text"] = data["free_text"].progress_apply(
            anonymize_names_with_transformers
        )
        logger.info("🫥 Annonymize with Transformer - do_better")
        data["do_better"] = data["do_better"].progress_apply(
            anonymize_names_with_transformers
        )

        logger.info("📗 Text Preprocesssing with *NLPretext")
        data["free_text"] = data["free_text"].apply(
            lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan
        )
        data["do_better"] = data["do_better"].apply(
            lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan
        )

        data = sentiment_analysis(data, "free_text")
        data = sentiment_analysis(data, "do_better")

        data = cleanup_neutral_sentiment(data, "free_text")
        data = cleanup_neutral_sentiment(data, "do_better")

        data = feedback_classification(data, batch_size=8)
        data = improvement_classification(data, batch_size=8)

        data = emotion_classification(data, "free_text", classifier=classifier)
        data = emotion_classification(data, "do_better", classifier=classifier)

        data = question_answering(data, "free_text")
        data = question_answering(data, "do_better")

        logger.info("Data pre-processing completed")

        concat_save_final_df(processed_data, data)

        do_git_merge()  # Push everything to GitHub
        logger.info("👍 Pushed to GitHub - Master Branch")
        logger.info("🎉 Successful Run completed")
    else:
        print(f"{Fore.RED}[*] No New rows to add - terminated.")
        logger.error("❌ Make Data terminated - No now rows")
