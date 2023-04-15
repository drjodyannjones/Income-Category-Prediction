# Use the official Python base image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY ./src .

# Expose the port on which the app will run
EXPOSE 8000

# Start the application using an ASGI server (Uvicorn in this case)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

