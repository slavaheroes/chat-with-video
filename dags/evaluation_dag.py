import os
import json
from datetime import datetime, timedelta

import openai
import pendulum
import psycopg2
from airflow import DAG
from airflow.operators.python import PythonOperator

client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)


JUDGE_PROMPT = '''
You are an impartial AI judge tasked with evaluating the quality and accuracy of an AI assistant's response to a user query.
1. User question: {user_question}
2. LLM Answer: {llm_answer}
3. Retrieved Context: {context}

Your task is to evaluate the LLM Answer and Retrieved Context based on the Relevance.
Answer in the following format:
 - relevance_score: An integer from 1-5 (1 being completely irrelevant, 5 being highly relevant)
 - explanation: A concise string explaining your scores and suggesting improvements. Keep this under 200 words.

You must provide your evaluation in the following JSON format:
{{"answer_relevance_score": <int>,
"context_relevance_score": <int>,
"explanation": "<string>"}}
'''


def get_db_connection():
    return psycopg2.connect(
        dbname=os.environ.get('POSTGRES_DB'),
        user=os.environ.get('POSTGRES_USER'),
        password=os.environ.get('POSTGRES_PASSWORD'),
        host=os.environ.get('POSTGRES_HOST'),
        port=os.environ.get('POSTGRES_PORT'),
    )


def get_unevaluated_conversations():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.conversation_id, c.question_id, c.question, c.answer, c.context
                FROM conversations c
                LEFT JOIN evaluation e ON c.question_id = e.question_id
                WHERE e.question_id IS NULL
            """
            )
            return cur.fetchall()


def evaluate_conversation(conversation):
    conversation_id, question_id, question, answer, context = conversation
    prompt = JUDGE_PROMPT.format(user_question=question, llm_answer=answer, context=context)

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating conversations."},
            {"role": "user", "content": prompt},
        ],
    )

    print(f'[INFO] Evaluation response: {response.choices[0].message.content}')
    try:
        evaluation = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f'[ERROR] Failed to parse evaluation response: {e}')
        evaluation = {
            'answer_relevance_score': 1,
            'context_relevance_score': 1,
            'explanation': 'Failed to parse evaluation response',
        }

    return conversation_id, question_id, evaluation


def insert_evaluation(evaluation_data):
    conversation_id, question_id, evaluation = evaluation_data
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation (conversation_id, question_id, answer_relevance_score, context_relevance_score, explanation, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (
                    conversation_id,
                    question_id,
                    evaluation['answer_relevance_score'],
                    evaluation['context_relevance_score'],
                    evaluation['explanation'],
                    datetime.now(pendulum.timezone('Asia/Tokyo')),
                ),
            )
        conn.commit()


def evaluate_conversations():
    conversations = get_unevaluated_conversations()
    for conversation in conversations:
        evaluation_data = evaluate_conversation(conversation)
        insert_evaluation(evaluation_data)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'evaluation_dag',
    default_args=default_args,
    description='Evaluate conversations every 20 minutes',
    schedule_interval=timedelta(minutes=20),
    start_date=pendulum.datetime(2023, 1, 1, tz="Asia/Tokyo"),
    catchup=False,
) as dag:

    evaluate_task = PythonOperator(
        task_id='evaluate_conversations',
        python_callable=evaluate_conversations,
    )

evaluate_task
