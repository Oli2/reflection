# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create /user directory and move the gcp_service_account.json file there
RUN mkdir -p /config && mv /app/gcp_service_account.json /config/gcp_service_account.json

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_gemini.txt

ENV GRADIO_SERVER_NAME="0.0.0.0"

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "cot_reflection_app.py"]