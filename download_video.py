"""
Download a public sports video using yt-dlp.
Usage:
python download_video.py --url <video_url> --out input_video.mp4

Example:
python download_video.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import subprocess
import argparse
import sys

# Use a REAL direct video URL here (replace anytime)
DEFAULT_URL = "https://www.youtube.com/watch?v=Kwu1yIC-ssg"


def download(url: str, output_path: str = "input_video.mp4"):
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_path,
        url,
    ]

    print(f"Downloading: {url}")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"Saved to: {output_path}")

    except subprocess.CalledProcessError:
        print("Download failed.")
        print("Possible reasons:")
        print("1. Invalid / unsupported URL")
        print("2. yt-dlp outdated")
        print("3. Video restricted/private")
        print("4. Internet issue")
        print("\nTry updating yt-dlp:")
        print("pip install -U yt-dlp")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Direct video URL")
    parser.add_argument("--out", default="input_video.mp4", help="Output filename")
    args = parser.parse_args()

    download(args.url, args.out)