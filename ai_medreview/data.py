import os
import pandas as pd
from transformers import pipeline
from sheethelper import SheetHelper
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (
    normalize_whitespace,
    remove_punct,
    remove_eol_characters,
    remove_stopwords,
    lower_text,
    replace_phone_numbers,
)
from nlpretext.social.preprocess import remove_mentions, remove_hashtag, remove_emoji
from tqdm import tqdm
tqdm.pandas()

from ai_medreview.params import *
from ai_medreview.utils import *
from ai_medreview.automation.git_merge import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")

from sheethelper import *

from loguru import logger

logger.add("log/debug.log", rotation="5000 KB")

# Select Classification Model - facebook/bart-large-mnli or FacebookAI/roberta-large-mnli
classification_model = "facebook/bart-large-mnli"


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
    ]
    data["time"] = pd.to_datetime(data["time"], format="%Y/%m/%d %H:%M:%S")
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
    for index, row in tqdm(data.iterrows()):
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

# Function to anonymize names in text


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
    logger.info("Calling Batch Generator")
    for i in range(0, len(data), batch_size):
        batch = data[column_name][i : i + batch_size]
        # Logging the batch content; you can comment this out or remove it in production
        yield batch, i  # Yield the batch and the starting index


@time_it
def feedback_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(classification_model).to(
        "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification", model=model, tokenizer=tokenizer, device=-1
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
    for batch, start_index in tqdm(batch_generator(data, "free_text", batch_size)):
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
    model = AutoModelForSequenceClassification.from_pretrained(classification_model).to(
        "cpu"
    )
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

    # Iterate over data in batches
    for batch, start_index in tqdm(batch_generator(data, "do_better", batch_size)):
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
        "Extremely likely": 5,
        "Likely": 4,
        "Neither likely nor unlikely": 3,
        "Unlikely": 2,
        "Extremely unlikely": 1,
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
    combined_data.to_parquet(f"{DATA_PATH}/data.parquet", index=False)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", encoding='utf-8', index=False)
    print(f"💾 data.csv saved to: {DATA_PATH}")


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


def text_preprocessing(text):
    logger.info("⭐️ Text Preprocesssing with *NLPretext")

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


if __name__ == "__main__":

    logger.info("▶️ Friends & Family Test Analysis - MAKE DATA - Started")

    # Load new data from Google Sheet
    raw_data = load_google_sheet()
    logger.info("Google Sheet data loaded")

    # Load local data.csv to dataframe
    processed_data = load_local_data()
    logger.info("Data.csv Loadded")

    # Return new data for processing
    data = raw_data[~raw_data.index.isin(processed_data.index)]
    logger.info(f"🆕 New rows to process: {data.shape[0]}")

    if data.shape[0] != 0:
        data = word_count(data)  # word count
        data = add_rating_score(data)

        data = clean_data(data)

        logger.info("🫥 Annonymize with Transformer")
        data["free_text"] = data["free_text"].progress_apply(anonymize_names_with_transformers)
        data["do_better"] = data["do_better"].progress_apply(anonymize_names_with_transformers)

        data["free_text"] = data["free_text"].progress_apply(
            lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan
        )
        data["do_better"] = data["do_better"].progress_apply(
            lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan
        )

        data = sentiment_analysis(data, "free_text")
        data = sentiment_analysis(data, "do_better")

        data = cleanup_neutral_sentiment(data, "free_text")
        data = cleanup_neutral_sentiment(data, "do_better")

        data = feedback_classification(data, batch_size=16)
        data = improvement_classification(data, batch_size=16)
        logger.info("Data pre-processing completed")

        concat_save_final_df(processed_data, data)

        do_git_merge()  # Push everything to GitHub
        logger.info("👍 Pushed to GitHub - Master Branch")
        logger.info("🎉 Successful Run completed")
    else:
        print(f"{Fore.RED}[*] No New rows to add - terminated.")
        logger.error("❌ Make Data terminated - No now rows")
