import os

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

SYS_PROMPT = '''
You are an intelligent AI assistant specialized in analyzing video transcripts and answering user questions about video content.
Your primary function is to help users quickly find and understand information from videos without having to watch them in full.
Always maintain a helpful, informative, and neutral tone.
'''

LLM_PROMPT = '''
You will be provided with a user's question and relevant transcript excerpts. Your task is to:

1. Carefully analyze the transcript to find the most relevant information.
2. Provide a clear, concise answer to the user's question based solely on the transcript content.
3. If the transcript doesn't contain enough information to fully answer the question, state this clearly and use your knowledge.
4. Include the timestamp or range from the transcript where you found the information.

Video Title: {title}

Author: {author}

Question: {question}

Retrieved Transcripts:
{transcript}

Answer the question as accurately as possible.
'''

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


def get_response(user_query, context, metadata):
    transcript = '\n'.join([item['text'] for item in context])
    title = metadata['title']
    author = metadata['author']

    prompt = LLM_PROMPT.format(title=title, author=author, question=user_query, transcript=transcript)

    llm_response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": SYS_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )

    # gpt-4o-mini fixed
    cost = (0.15 * llm_response.usage.prompt_tokens + 0.06 * llm_response.usage.completion_tokens) / 1000

    return {
        **metadata,
        'prompt': prompt,
        'context': transcript,
        'answer': llm_response.choices[0].message.content,
        'cost': cost,
        'prompt_tokens': llm_response.usage.prompt_tokens,
        'completion_tokens': llm_response.usage.completion_tokens,
    }
