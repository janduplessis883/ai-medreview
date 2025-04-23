from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pydantic import BaseModel
import torch

# Initialize FastAPI app
app = FastAPI(
    title="GP Surgery Review Classifier API",
    description="Classifies GP Surgery Review Strings into predefined categories using zero-shot classification",
    version="1.0.0",
)

# Model Initialization (moved outside the API function for efficiency)
try:
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")#.to("cuda" if torch.cuda.is_available() else 'cpu') #move model on cuda if available
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available, else CPU
        framework="pt",
    )

    print("Model and Tokenizer loaded successfully!")  #Confirmation
except Exception as e:
    print(f"Error initializing model and tokenizer: {e}")
    raise  # Raise the exception to prevent the API from starting if initialization fails


# Define categories for classification
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
]  # Include your categories here


# Pydantic Model for Request Body
class ReviewRequest(BaseModel):
    review_text: str


# Pydantic Model for Response Body
class ClassificationResponse(BaseModel):
    category: str
    confidence_score: float  # Add confidence score


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classifies a GP Surgery Review String",
    description="Receives a GP Surgery Review string and returns the most relevant category.",
)
async def classify_review(request: ReviewRequest):
    """
    Classifies a GP Surgery Review String into predefined categories.

    Args:
      item: A dictionary containing the review text.

    Returns:
      A dictionary containing the most relevant category.
    """
    try:
        result = classifier(request.review_text, categories)
        # Return the top category with associated score
        return ClassificationResponse(category=result["labels"][0], confidence_score=result["scores"][0])
    except Exception as e:
        print(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))  # Return error to the client



if __name__ == "__main__":
    # Example Usage (for testing purposes)
    import uvicorn

    # Running the application using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
