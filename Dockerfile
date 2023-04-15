# Use the official Python base image
FROM python:3.9

# Set the working directory to /src
WORKDIR /src

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY ./src /src

# Expose the port on which the app will run
EXPOSE 8000

# Start the application using an ASGI server (Uvicorn in this case)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
