# Use an official Python image
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /usr/src/ml_service

# Install system dependencies including compilers
RUN apk add --no-cache \
    gcc \
    g++ \
    make

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run the machine learning script
CMD ["python", "ml_main.py"]
