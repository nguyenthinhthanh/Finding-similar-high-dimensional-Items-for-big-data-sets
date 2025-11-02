# Using Python 3.13 as the base image
FROM python:3.13

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt /app/

# Install dependencies and clean up for efficiency
RUN apt-get update && apt-get install -y build-essential \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && apt-get remove -y build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy the entire application source code into the container.
COPY app/src /app/src
COPY benchmarks /app/benchmarks

ENV PYTHONPATH=/app/src

# Default command (can be overridden in docker-compose)
CMD ["bash"]
