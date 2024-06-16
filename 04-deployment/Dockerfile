# Use the specified base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Update pip
RUN pip install -U pip

# Install pipenv
RUN pip install pipenv

# Set the working directory to /app
WORKDIR /app

# Copy Pipfile and Pipfile.lock to the working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies from Pipfile
RUN pipenv install --system --deploy

# Copy the starter.py script to the working directory
COPY ["starter_2.py", "./"]

# Set the default command to run the script
ENTRYPOINT ["python", "starter_2.py"]
