import os
import re

from dotenv import load_dotenv

load_dotenv('../.env')  # for repdoducibility change to ../env.env

import json
from typing import Dict, List, Tuple

import openai
import pandas as pd
from tqdm import tqdm
from loguru import logger
from pytube import YouTube
from prompts import GENERATION_PROMPT
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

VIDEOS = {
    'SamAltman_interview': 'https://youtu.be/jvqFAi7vkBc?si=qAfRepbBUUEXhJaP',  # interview
    'vDud_Kolyma': 'https://www.youtube.com/watch?v=oo1WouI38rQ',  # documentary
    'PavelDurov_interview': 'https://www.youtube.com/watch?v=1Ut6RouSs0w',  # interview
}


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
        logger(f"Error: {e}")
        return None


def load_transcript(video_id: str) -> List[dict]:
    '''
    Load the transcript of the video
    '''
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        logger(f"Error: {e}")
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


client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)


def llm(prompt: str, max_tokens: int = 1000) -> str:
    '''
    Use the language model to generate questions
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.5,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates questions and answers based on provided text.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    for video_name, video_url in VIDEOS.items():
        valid, video_id = get_video_id(video_url)
        if valid:
            metadata = get_metadata(video_id)
            if metadata:
                logger.info(f"Metadata for video: {video_name}")
                logger.info(f"Author: {metadata['author']}")
                logger.info(f"Title: {metadata['title']}")
                logger.info(f"Description: {metadata['description']}")

            trasnscript = load_transcript(video_id)
            if trasnscript:
                logger.success(f"Transcript loaded for video: {video_name}")

            chunks = process_transcript(trasnscript)
            logger.info(f"Number of chunks: {len(chunks)}")

            data = []

            for chunk_id, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Processing video: {video_name}"):
                prompt = GENERATION_PROMPT.format(text=chunk.page_content)

                questions_json = llm(prompt)

                try:
                    qa_pairs = json.loads(questions_json)
                    for qa in qa_pairs:
                        data.append(
                            {
                                'chunk_id': chunk_id,
                                'chunk': chunk.page_content,
                                'llm_output': questions_json,
                                'question': qa['question'],
                                'answer': qa['answer'],
                            }
                        )

                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue

            data = pd.DataFrame(data)
            data.to_csv(f'data/{video_name}_gt.csv', index=False)

            logger.success(f"Data saved for video: {video_name}")

    logger.success("All data saved!")
