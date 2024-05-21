# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the environment variables
ENV PYTHONUNBUFFERED 1

# Run the specified Python script when the container starts
CMD ["python", "ai_medreview/scheduler.py"]