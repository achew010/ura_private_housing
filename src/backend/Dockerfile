# Use an official Python image
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY . .

# Run the Python script
CMD ["python", "main.py"]
