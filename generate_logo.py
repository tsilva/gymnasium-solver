#!/usr/bin/env python3
"""
Generate logo for gymsolve using OpenAI DALL-E 3.
Based on project-logo-author configuration style.
"""

import os
import sys
import base64
from pathlib import Path
from io import BytesIO


def main():
    try:
        from openai import OpenAI
        from PIL import Image
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: uv add openai pillow")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Logo configuration
    project_name = "gymsolve"
    key_color = "#00FF00"

    # Prompt based on project-logo-author config (SNES style, pixel art)
    prompt = f"""SNES 16-bit pixel art style logo for "{project_name}" (a reinforcement learning framework for Gymnasium).

Design elements:
- Main subject: A charming robot character mascot representing an AI agent solving gym environments, in Chrono Trigger SNES pixel art style
- Robot should have bright saturated colors: vivid oranges, electric cyans, deep blues, purples, golden yellows (NO green tones on the character)
- VISIBLE CHUNKY PIXELS with dithering for shading and selective black outlines
- Floating icon symbols around the robot (no text on these icons): neural network nodes, retro game controller, trophy
- Bottom banner with "{project_name.upper()}" in pixel art text, bright cyan and purple gradient
- Pure bright green {key_color} background - the robot and icons must NOT use green tones
- Full SNES color palette, bright and saturated like LucasArts adventure games
- High contrast, eye-catching, maximum saturation and vibrancy
- Square format, centered composition

Style: 16-bit pixel art, retro gaming aesthetic, 1990s SNES RPG style"""

    print(f"Generating logo for {project_name} using DALL-E 3...")
    print(f"Style: SNES 16-bit pixel art (Chrono Trigger style)")
    print(f"Background: {key_color} (bright green)")
    print(f"Text: {project_name.upper()}")

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json",
        )

        # Decode and save the image
        image_data = base64.b64decode(response.data[0].b64_json)
        image = Image.open(BytesIO(image_data))

        output_path = Path("logo.png")
        image.save(output_path, "PNG")

        print(f"\n✓ Logo saved to: {output_path.absolute()}")
        print(f"  Size: {image.size[0]}x{image.size[1]} pixels")

    except Exception as e:
        print(f"Error generating logo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
