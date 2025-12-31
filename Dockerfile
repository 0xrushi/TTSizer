FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git wget curl vim nano

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first (layer caching optimization)
# uv sync will only run when these files change
COPY pyproject.toml README.md ./

# Copy ttsizer/ directory needed for hatchling to build the package
COPY ttsizer/ ./ttsizer/

# Install dependencies using uv sync
# Packages already installed in base image (torch, torchaudio, torchvision) will be skipped
# Packages configured in [tool.uv.sources] will use their specified sources
# Note: flash-attn will be built from source (takes ~10-15 minutes)
RUN uv sync

# Copy rest of the project (code changes won't trigger uv sync)
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY assets/ ./assets/
COPY main.py ./
COPY Makefile ./

# Expose port if needed
EXPOSE 8888

# Set default command
CMD ["/bin/bash"]
