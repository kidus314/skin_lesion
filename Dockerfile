# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the model and code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
