from cog import Input, Path
import moderngl
import numpy as np
from PIL import Image
import random
import re
import cv2
import subprocess
import os


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to normalized RGB values (0.0-1.0)"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def goo(
    seed: int = Input(
        default=-1,
        description="Seed for the random number generator",
    ),
    width: int = Input(
        default=1024,
        description="Dimension of the output image",
        ge=1,
        le=4096,
    ),
    height: int = Input(
        default=1024,
        description="Height of the output image",
        ge=1,
        le=4096,
    ),
    color1: str = Input(
        default="#F38020",
        description="First color (hex format)",
    ),
    color2: str = Input(
        default="#F48120",
        description="Second color (hex format)",
    ),
    color3: str = Input(
        default="#FAAD3F",
        description="Third color (hex format)",
    ),
    scale: int = 1,
    depth: int = 3,
    format: str = Input(
        description="Format of the output (image or video)",
        choices=["png", "jpeg", "tiff", "mp4"],
        default="mp4",
    ),
    speed: float = Input(
        default=2.0,
        description="Speed of the goo animation effect",
        ge=0.0,
        le=10.0,
    ),
    num_frames: int = Input(
        default=60 * 10,
        description="Number of frames for video output (only used when format is mp4)",
        ge=1,
        le=600,
    ),
    fps: int = Input(
        default=60,
        description="Frames per second for video output (only used when format is mp4)",
        ge=1,
        le=60,
    ),
    pingpong: bool = Input(
        default=True,
        description="Play video forward then backward (only used when format is mp4)",
    ),
) -> Path:
    if seed == -1:
        seed = int(random.random() * 2**16)

    ctx = moderngl.create_context(
        standalone=True,
        backend="egl",
    )

    prog = ctx.program(
        vertex_shader="""
        #version 330
        in vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """,
        fragment_shader=f"""
        #version 330
        precision mediump float;
        uniform vec2 iResolution;
        uniform float iTime;
        uniform vec2 iMouse;
        uniform vec3 color1;
        uniform vec3 color2;
        uniform vec3 color3;

        vec2 effect(vec2 p, float i, float time) {{
            vec2 mouse = vec2(0.0, 0.0); // Ignoring mouse input as per instructions
            return vec2(sin(p.x * i + time) * cos(p.y * i + time), sin(length(p.x)) * cos(length(p.y)));
        }}

        void main() {{
            vec2 p = (2.0 * gl_FragCoord.xy - iResolution.xy) / max(iResolution.x, iResolution.y);
            p.x += {seed:.1f}; // Use the seed prop to offset the starting position of the goo effect
            p.y += {seed:.1f};

            p *= {scale:.1f};
            for (int i = 1; i < {depth}; i++) {{
                float fi = float(i);
                p += effect(p, fi, iTime * ({speed:.1f}/10));
            }}
            vec3 col = mix(mix(color1, color2, 1.0-sin(p.x)), color3, cos(p.y+p.x));
            gl_FragColor = vec4(col, 1.0);
        }}
    """,
    )

    vertices = np.array(
        [
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
        ],
        dtype="f4",
    )

    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, "position")
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)])

    # Convert hex colors to RGB and set uniforms
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    rgb3 = hex_to_rgb(color3)
    prog["color1"].value = rgb1
    prog["color2"].value = rgb2
    prog["color3"].value = rgb3

    if format == "mp4":
        # Set up FFmpeg process for MP4/H.264 encoding
        temp_filename = "/tmp/output_temp.mp4" if pingpong else "/tmp/output.mp4"
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'fast',  # Encoding preset
            '-profile:v', 'baseline',  # Most compatible H.264 profile
            '-pix_fmt', 'yuv420p',  # Required for browser compatibility
            '-movflags', '+faststart',  # Enable streaming
            '-crf', '23',  # Quality setting (lower = better, 23 is a good default)
            temp_filename
        ]

        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Generate frames
            for frame in range(num_frames):
                fbo.use()
                ctx.clear()
                iResolution = (width, height)
                iTime = frame / fps
                prog["iResolution"].value = iResolution
                prog["iTime"].value = iTime
                vao.render(moderngl.TRIANGLES)

                data = fbo.read(components=3)
                image = Image.frombytes("RGB", fbo.size, data)
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

                # Write raw frame data directly to FFmpeg's stdin
                ffmpeg_process.stdin.write(image.tobytes())

            # Close stdin and wait for FFmpeg to finish
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

            if ffmpeg_process.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed with error: {ffmpeg_process.stderr.read().decode()}")

        finally:
            # Ensure resources are cleaned up
            if ffmpeg_process.poll() is None:
                ffmpeg_process.terminate()
                ffmpeg_process.wait()

        # Apply pingpong effect if enabled
        if pingpong:
            final_filename = "/tmp/output.mp4"
            subprocess.run([
                "ffmpeg",
                "-i", temp_filename,
                "-filter_complex", "[0:v]reverse[r];[0:v][r]concat=n=2:v=1[v]",
                "-map", "[v]",
                "-an",  # Remove audio
                "-c:v", "libx264",
                "-preset", "fast",
                "-profile:v", "baseline",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-crf", "23",
                "-y", final_filename
            ], check=True)

            # Clean up temporary file
            os.remove(temp_filename)
            return Path(final_filename)
        else:
            return Path(temp_filename)
    else:
        # Original image generation code
        fbo.use()
        ctx.clear()
        iResolution = (width, height)
        iTime = 1  # Modulo to prevent too large numbers
        prog["iResolution"].value = iResolution
        prog["iTime"].value = iTime
        vao.render(moderngl.TRIANGLES)  # pylint: disable=no-member

        data = fbo.read(components=3)
        image = Image.frombytes("RGB", fbo.size, data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # pylint: disable=no-member

        # Save the output image
        ext = re.sub(r"\W+", "", format)
        if format == "jpeg":
            image = image.convert("RGB")
            ext = "jpg"
        filename = f"/tmp/output.{ext}"
        image.save(filename, format=format)

        return Path(filename)
