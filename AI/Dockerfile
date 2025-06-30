FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/AI:$PYTHONPATH 

ARG AI_PORT
ENV AI_PORT=${AI_PORT}

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt --timeout=120

# Copy the rest of the application code into the container
COPY . .

# Expose the port for the AI service
EXPOSE ${AI_PORT}

# Command to run the AI service
CMD ["python", "app.py"]
