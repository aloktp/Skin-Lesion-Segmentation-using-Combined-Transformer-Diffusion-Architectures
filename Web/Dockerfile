FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Define build arguments for host and port
ARG WEB_HOST
ARG WEB_PORT
ARG FLASK_APP=main.py
ENV FLASK_APP=${FLASK_APP}
ENV WEB_HOST=${WEB_HOST}
ENV WEB_PORT=${WEB_PORT}

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt --timeout=120

# Copy the rest of the application code into the container
COPY . .

# Expose the port for the Web service
EXPOSE ${WEB_PORT}

# Command to run the Flask app
CMD ["sh", "-c", "flask run --host=${WEB_HOST} --port=${WEB_PORT}"]
