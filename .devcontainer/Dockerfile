FROM dolfinx/dolfinx:v0.9.0

# Install system utilities and mesh generation core
RUN apt-get update && \
      apt-get install -y --no-install-recommends sudo xvfb libgl1 libglu1-mesa && \
      rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt (single source of truth)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# spilsnet-torch (SPILS-Net architecture) is installed as part of requirements.txt.

# Set the working directory
WORKDIR /workspace
