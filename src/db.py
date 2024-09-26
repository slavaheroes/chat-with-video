import os
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import DictCursor

# Load environment variables
load_dotenv('.env')

# Database connection parameters
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")

# Define UTC+9 timezone
UTC_PLUS_9 = ZoneInfo("Asia/Tokyo")


def get_current_time_utc9():
    return datetime.now(UTC_PLUS_9)


def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)


def create_tables():
    conn = get_db_connection()
    cur = conn.cursor()

    # Create conversations table
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT,
            question_id TEXT,
            title TEXT,
            author TEXT,
            description TEXT,
            video_id TEXT,
            prompt TEXT,
            question TEXT,
            context TEXT,
            answer TEXT,
            cost FLOAT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
    )

    # Create feedback table
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT,
            feedback INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
    )

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS evaluation (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT,
            question_id TEXT,
            question TEXT,
            answer_relevance INT,
            context_relevance INT,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
    )

    conn.commit()
    cur.close()
    conn.close()


def save_conversation(conversation):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        '''
        INSERT INTO conversations (
            conversation_id, question_id, title, author, description, video_id,
            prompt, question, context, answer, cost,
            prompt_tokens, completion_tokens, latency, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''',
        (
            conversation['conversation_id'],
            conversation.get('question_id', ''),
            conversation.get('title', ''),
            conversation.get('author', ''),
            conversation.get('description', ''),
            conversation.get('video_id', ''),
            conversation.get('prompt', ''),
            conversation['question'],
            conversation.get('context', ''),
            conversation['answer'],
            conversation.get('cost', 0),
            conversation.get('prompt_tokens', 0),
            conversation.get('completion_tokens', 0),
            conversation['latency'],
            get_current_time_utc9(),
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


def save_feedback(conversation_id, feedback):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        '''
        INSERT INTO feedback (conversation_id, feedback, created_at)
        VALUES (%s, %s, %s)
    ''',
        (conversation_id, feedback, get_current_time_utc9()),
    )

    conn.commit()
    cur.close()
    conn.close()


def get_recent_conversations(limit=3, conversation_id=None):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    if conversation_id:
        cur.execute(
            '''
            SELECT * FROM conversations
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        ''',
            (conversation_id, limit),
        )
    else:
        cur.execute(
            '''
            SELECT * FROM conversations
            ORDER BY created_at DESC
            LIMIT %s
        ''',
            (limit,),
        )

    conversations = cur.fetchall()

    cur.close()
    conn.close()

    return [dict(conv) for conv in conversations]


# Create tables when the module is imported
create_tables()
