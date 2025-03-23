# Use a lightweight base image with Python
FROM python:3.9-slim AS base

# Set the working directory
WORKDIR /app

# Copy only the requirements to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY . /app

# Expose the port Gunicorn will use
EXPOSE 5005

ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Define the command to run the application with Gunicorn
# CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5005", "app:app"]
# CMD ["gunicorn", "-w", "2", "-k", "sync", "-b", "0.0.0.0:5005", "--threads", "4", "app:app"]
# Use Gunicorn as the WSGI server
CMD ["sh", "-c", "gunicorn -w ${GUNICORN_WORKERS:-2} -k sync -b 0.0.0.0:5005 --threads 4 app:app"]


