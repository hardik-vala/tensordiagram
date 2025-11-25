#!/usr/bin/env python3
"""
quick programmatic test example for tensordiagram MCP server.
"""
import asyncio
import base64
from PIL import Image
import io

from fastmcp import Client
from server import mcp


async def main():
    client = Client(mcp)

    async with client:
        print("testing draw_tensor tool...")

        result = await client.call_tool(
            "draw_tensor",
            arguments={
                "shape": [3, 4, 10],
                "color": "lightblue",
                "show_dim_sizes": True
            }
        )

        img_data = base64.b64decode(result.content[0].data)
        img = Image.open(io.BytesIO(img_data))

        print(f"✓ generated image: {img.width}x{img.height}px")

        with open("quick_test_output.png", "wb") as f:
            f.write(img_data)

        print("✓ saved to: quick_test_output.png")


if __name__ == "__main__":
    asyncio.run(main())
