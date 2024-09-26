import re
from typing import Dict, List, Tuple

from loguru import logger
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_video_id(url: str) -> Tuple[bool, str]:
    '''
    Determine if the url is a valid youtube video url
    Return the video id if valid
    '''
    pattern = r'^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w\-_]{11})(?:\S+)?$'
    match = re.match(pattern, url)
    if match:
        return True, match.group(1)
    else:
        return False, None


def get_metadata(video_id: str) -> Dict:
    '''
    Get metadata of the video
    '''
    try:
        metadata = YouTube(f'https://www.youtube.com/watch?v={video_id}')

        return {
            'author': metadata.author if metadata.author else 'Unavailable',
            'title': metadata.title if metadata.title else 'Unavailable',
            'description': metadata.description if metadata.description else 'Unavailable',
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {'author': 'Unavailable', 'title': 'Unavailable', 'description': 'Unavailable'}


def load_transcript(video_id: str) -> List[dict]:
    '''
    Load the transcript of the video
    '''
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def process_transcript(transcript: List[dict]) -> List[str]:
    '''
    Process the transcript into chunks of text
    '''
    strings = []
    for item in transcript:
        strings.append(f"{item['start']}: {item['text']}")

    strings = '\n'.join(strings)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([strings])

    return chunks
