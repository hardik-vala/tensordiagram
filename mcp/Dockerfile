# tensordiagram MCP Server - Docker Image

FROM python:3.11-slim

# install system dependencies for cairo (required for PNG rendering)
RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy only MCP server files
COPY server.py .
COPY __init__.py .
COPY pyproject.toml .
COPY README.md .
COPY DESIGN.md .

# install dependencies
RUN pip install --no-cache-dir \
    tensordiagram>=0.1.2 \
    fastmcp>=2.0.0 \
    pycairo \
    numpy

# make server.py executable
RUN chmod +x server.py

# expose port for HTTP/SSE transport
EXPOSE 8000

# environment variable for unbuffered Python output (better logging)
ENV PYTHONUNBUFFERED=1

# default: run in HTTP/SSE mode on port 8000
CMD ["python", "server.py", "--transport", "sse", "--port", "8000"]

# for stdio mode (when using with local MCP clients), override with:
# docker run -i tensordiagram-mcp python server.py
