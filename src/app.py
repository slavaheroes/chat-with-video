import time
import uuid

import streamlit as st
from loguru import logger

import db
import llm
import utils
import retrieval


def reset_app_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def main():
    st.set_page_config(page_title="Chat with Video", page_icon=":tv:")
    st.title("Chat with YouTube Video | LLM Zoomcamp 2024")

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    if 'index' not in st.session_state:
        st.session_state.index = None

    if 'metadata' not in st.session_state:
        st.session_state.metadata = None

    if not st.session_state.processed:
        with st.form(key='video_form'):
            video_link = st.text_input("Enter the video link:")
            process_button = st.form_submit_button("Process")

            if process_button:
                if video_link:
                    logger.info(f"Processing video: {video_link}")

                    is_valid, video_id = utils.get_video_id(video_link)
                    if is_valid:
                        metadata = utils.get_metadata(video_id)
                        metadata['video_id'] = video_id
                        st.write(f"Title: {metadata['title']}")
                        st.write(f"Author: {metadata['author']}")
                        st.session_state.metadata = metadata

                        try:
                            transcript = utils.load_transcript(video_id)
                            chunks = utils.process_transcript(transcript)
                            chunks = [chunk.page_content for chunk in chunks]
                            index = retrieval.tf_idf(chunks)
                            st.session_state.index = index
                            st.session_state.processed = True
                            st.session_state.conversation_id = str(uuid.uuid4())

                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            st.error("An error occurred while processing the video.")
                            st.stop()
                else:
                    st.error("Please enter a valid video link.")
    elif st.session_state.processed and st.session_state.index:

        st.success("Video processed successfully!")

        if "count" not in st.session_state:
            st.session_state.count = 0
            logger.info("Feedback count initialized to 0")

        # Chat interface
        st.subheader("Chat")
        user_input = st.text_input("Enter your question:")

        if st.button("Ask"):
            logger.info(f"User asked: '{user_input}'")
            with st.spinner("Processing..."):
                start = time.time()
                retrieved_docs = st.session_state.index.search(user_input, num_results=3)
                response = llm.get_response(user_input, retrieved_docs, st.session_state.metadata)
                llm_output = response['answer']
                time.sleep(1)
                end = time.time()
                st.success(f"Processed in {end - start:.2f} seconds")
                latency = end - start
                st.write(f"Answer: {llm_output}")

                conversation = {
                    'conversation_id': st.session_state.conversation_id,
                    'latency': latency,
                    'question_id': str(uuid.uuid4()),
                    'question': user_input,
                    **response,
                }

                db.save_conversation(conversation)

        # Feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(":thumbsup:"):
                logger.info("Positive feedback +1")
                st.session_state.count += 1
                db.save_feedback(st.session_state.conversation_id, 1)
        with col2:
            if st.button(":thumbsdown:"):
                logger.info("Negative feedback -1")
                st.session_state.count -= 1
                db.save_feedback(st.session_state.conversation_id, -1)

        st.write(f"Current count: {st.session_state.count}")
        # Display recent conversations
        st.subheader("Recent Conversations")

        recent_conversations = db.get_recent_conversations(limit=3, conversation_id=st.session_state.conversation_id)
        for conv in recent_conversations:
            st.write(f"Q: {conv['question']}")
            st.write(f"A: {conv['answer']}")
            st.write("---")

        # New Conversation button
        if st.button("New Conversation"):
            logger.info("Starting new conversation")
            reset_app_state()
            st.rerun()

    logger.info('Streamlit app loop finished.')


if __name__ == "__main__":
    main()
