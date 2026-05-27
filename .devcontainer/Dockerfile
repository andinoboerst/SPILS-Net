FROM dolfinx/dolfinx:v0.9.0

# Install system utilities and mesh generation core
RUN apt-get update && \
      apt-get install -y --no-install-recommends sudo xvfb libgl1 libglu1-mesa && \
      rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt (single source of truth)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -f https://data.pyg.org/whl/torch-2.2.0+cpu.html -r /tmp/requirements.txt

# spilsnet-torch is provided by the mounted host repository at runtime
# via docker-compose volume /spilsnet-torch and PYTHONPATH.

# Set the working directory
WORKDIR /workspace
