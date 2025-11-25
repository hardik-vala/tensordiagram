# Testing the TensorDiagram MCP Server

This guide explains how to test the tensordiagram MCP server locally and programmatically.

## Methods for Testing

### 1. Programmatic Testing with FastMCP Client (Recommended)

The `test_server.py` script demonstrates comprehensive testing using the FastMCP Client:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the test suite
python3 test_server.py
```

**Features tested:**
- ✓ Small tensors (width scaling to 400px minimum)
- ✓ Square tensors with color
- ✓ Wide tensors (no unnecessary scaling)
- ✓ 3D tensors
- ✓ 1D tensors with values
- ✓ 2D tensors with annotations
- ✓ Error handling for invalid inputs

**Output:**
- Console output showing test results
- Sample images saved to `test_output/` directory

### 2. Interactive Development Mode

Use FastMCP's development server for interactive testing:

```bash
source .venv/bin/activate
fastmcp dev server.py
```

This opens an interactive inspector where you can:
- View available tools
- Test tool calls with custom parameters
- See real-time responses
- Debug issues

### 3. Manual Testing via STDIO

Run the server directly for integration testing with MCP clients:

```bash
source .venv/bin/activate
python3 server.py
```

The server runs in STDIO mode by default, which is what Claude Desktop and other MCP clients use.

### 4. HTTP/SSE Mode for Web Testing

For testing with HTTP clients:

```bash
source .venv/bin/activate
python3 server.py --transport sse --port 8000
```

Then connect to `http://localhost:8000` with your HTTP client.

## Writing Custom Tests

Here's a minimal example of programmatic testing:

```python
import asyncio
import base64
from PIL import Image
import io
from fastmcp import Client
from server import mcp

async def simple_test():
    client = Client(mcp)

    async with client:
        # Call the draw_tensor tool
        result = await client.call_tool(
            "draw_tensor",
            arguments={"shape": [3, 4], "color": "lightblue"}
        )

        # Decode and inspect the image
        img_data = base64.b64decode(result.content[0].data)
        img = Image.open(io.BytesIO(img_data))

        print(f"Generated image: {img.width}x{img.height}px")

        # Save to file
        with open("output.png", "wb") as f:
            f.write(img_data)

asyncio.run(simple_test())
```

## Key Testing Points

### Minimum Width Requirement

The server ensures all output images have a minimum width of 400px:

- Images narrower than 400px are scaled up (maintaining aspect ratio)
- Images already 400px or wider are not modified
- Aspect ratio is always preserved

Test this with:
```python
# Small tensor - will be scaled
result = await client.call_tool("draw_tensor", arguments={"shape": [2, 3]})
# Expected: ~400px width

# Wide tensor - no scaling needed
result = await client.call_tool("draw_tensor", arguments={"shape": [2, 20]})
# Expected: ~2000px width (no change)
```

### Error Handling

Test error cases:
```python
# Invalid: 4D tensor (not supported)
await client.call_tool("draw_tensor", arguments={"shape": [2, 2, 2, 2]})

# Invalid: values length mismatch
await client.call_tool("draw_tensor", arguments={
    "shape": [2, 3],
    "values": [1, 2, 3]  # Should be 6 values
})
```

## Resources

- [FastMCP Documentation](https://gofastmcp.com/)
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Protocol](https://modelcontextprotocol.io/)
