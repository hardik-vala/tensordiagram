# tensordiagram mcp server

mcp server that enables AI assistants like Claude to draw diagrams of tensors.

## overview

This mcp server provides the `draw_tensor` tool, allowing LLMs to create clear, visual representations of tensors to help users understand tensor shapes, operations, and dimensional transformations. Built on top of [tensordiagram](https://github.com/hardik-vala/tensordiagram) and [FastMCP](https://github.com/jlowin/fastmcp).

## install

### option 1: PyPI

```bash
pip install tensordiagram-mcp
```

### option 2: from source

```bash
git clone https://github.com/hardik-vala/tensordiagram.git
cd tensordiagram/mcp
pip install -e .
```

### option 3: Docker

```bash
git clone https://github.com/hardik-vala/tensordiagram.git
cd tensordiagram/mcp
docker build -t tensordiagram/mcp:latest .
docker run -i --rm --name tensordiagram-mcp tensordiagram/mcp:latest
```

## configuration

### Claude Desktop

Add this to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

#### using pip installation

```json
{
  "mcpServers": {
    "tensordiagram": {
      "command": "python",
      "args": ["-m", "tensordiagram_mcp"]
    }
  }
}
```

#### using local installation

```json
{
  "mcpServers": {
    "tensordiagram": {
      "command": "python",
      "args": ["/path/to/tensordiagram/mcp/server.py"]
    }
  }
}
```

#### using Docker

```json
{
  "mcpServers": {
    "tensordiagram": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "--name", "tensordiagram-mcp", "tensordiagram/mcp:latest"]
    }
  }
}
```

After updating the configuration, restart Claude Desktop.

## usage

Once installed, Claude can use the `draw_tensor` tool to create visualizations. Here are some examples:

### example prompts

> "Can you show me what a 3x4 tensor looks like?"
> "Draw a 2x3x4 tensor in coral color"
> "Visualize a 3x3 identity matrix with the actual values shown"
> "Create a diagram of a 4x6 tensor with dimension sizes and indices labeled"

## technical details

- **minimum width**: all output images are scaled to at least 400px width for readability
- **size limit**: images are capped at ~900KB to comply with MCP's 1MB message limit
- **image format**: PNG with white background
- **dependencies**: requires pycairo for PNG rendering

## license

MIT License - see the main tensordiagram repository for details.
