import time
import uuid

import pandas as pd
from loguru import logger

import db
import llm
import utils
import retrieval


def simulation():
    VIDEOS = {
        'SamAltman_interview': 'https://youtu.be/jvqFAi7vkBc?si=qAfRepbBUUEXhJaP',  # interview
        'vDud_Kolyma': 'https://www.youtube.com/watch?v=oo1WouI38rQ',  # documentary
        'PavelDurov_interview': 'https://www.youtube.com/watch?v=1Ut6RouSs0w',  # interview
    }

    for video_name, video_url in VIDEOS.items():
        logger.info(f"Processing video: {video_name}")
        _, video_id = utils.get_video_id(video_url)
        metadata = utils.get_metadata(video_id)
        metadata['video_id'] = video_id
        transcript = utils.load_transcript(video_id)
        chunks = utils.process_transcript(transcript)
        chunks = [chunk.page_content for chunk in chunks]
        index = retrieval.tf_idf(chunks)

        questions = pd.read_csv(f'data/{video_name}_gt.csv')['question'].sample(10)
        conversation_id = str(uuid.uuid4())
        for q_id, user_query in enumerate(questions):
            start = time.time()
            retrieved_docs = index.search(user_query, num_results=3)
            response = llm.get_response(user_query, retrieved_docs, metadata)
            end = time.time()
            latency = end - start
            conversation = {
                'conversation_id': conversation_id,
                'latency': latency,
                'question_id': str(uuid.uuid4()),
                'question': user_query,
                **response,
            }
            db.save_conversation(conversation)

            if q_id % 2 == 0:
                # positive feedback
                db.save_feedback(conversation_id, feedback=1)
            else:
                # negative feedback
                db.save_feedback(conversation_id, feedback=-1)

        logger.info(f"Finished processing video: {video_name}")


simulation()
