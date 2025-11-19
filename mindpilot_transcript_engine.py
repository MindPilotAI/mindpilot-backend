import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


def fetch_transcript_text(video_id: str) -> str:
    """
    Fetches the transcript for a given video ID and returns it as one combined string.
    Uses the new YouTubeTranscriptApi().fetch(...) interface.
    """
    api = YouTubeTranscriptApi()

    try:
        # Try to fetch English transcript first (you can adjust languages as needed)
        fetched = api.fetch(video_id, languages=['en'])
        # fetched is a FetchedTranscript object; convert to raw list of dicts
        raw_chunks = fetched.to_raw_data()
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {e}")

    # Join all text chunks into one string
    full_text = " ".join(chunk.get("text", "") for chunk in raw_chunks)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text



def extract_video_id(youtube_url: str) -> str:
    """
    Extracts the YouTube video ID from various URL formats.
    Examples:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    """
    parsed = urlparse(youtube_url)

    # Case 1: standard watch URL
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

    # Case 2: shortened youtu.be URL
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    # Fallback: try to pull a video-like ID via regex
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")


def fetch_transcript_text(video_id: str) -> str:
    """
    Fetches the transcript for a given video ID and returns it as one combined string.
    Uses the new YouTubeTranscriptApi().fetch(...) interface.
    """
    api = YouTubeTranscriptApi()

    try:
        # Try to fetch English transcript first (you can adjust languages as needed)
        fetched = api.fetch(video_id, languages=['en'])
        # fetched is a FetchedTranscript object; convert to raw list of dicts
        raw_chunks = fetched.to_raw_data()
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {e}")

    # Join all text chunks into one string
    full_text = " ".join(chunk.get("text", "") for chunk in raw_chunks)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


def save_text_to_file(text: str, output_path: str) -> None:
    """
    Saves the given text to a UTF-8 encoded .txt file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    youtube_url = input("Enter YouTube URL: ").strip()

    try:
        video_id = extract_video_id(youtube_url)
        print(f"Extracted video ID: {video_id}")
    except ValueError as e:
        print(e)
        return

    try:
        transcript_text = fetch_transcript_text(video_id)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    print("\nFirst 500 characters of transcript:\n")
    print(transcript_text[:500] + ("..." if len(transcript_text) > 500 else ""))

    output_file = "mindpilot_transcript_output.txt"
    save_text_to_file(transcript_text, output_file)
    print(f"\nFull transcript saved to: {output_file}")


if __name__ == "__main__":
    main()
