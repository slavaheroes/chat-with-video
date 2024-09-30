import os
import argparse

from dotenv import load_dotenv

load_dotenv('../.env')

import openai
import pandas as pd
import metrics
import anthropic
import minsearch
from tqdm import tqdm
from loguru import logger
from prompts import LLM_PROMPT, SYS_PROMPT, LLM_JUDGE_PROMPT
from generate_datasets import VIDEOS
from sentence_transformers import SentenceTransformer, util

anthropic_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY'),
)

openai_client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)


def haiku(prompt: str) -> str:
    '''
    Call the Claude-haiku
    '''
    response = anthropic_client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1024,
        temperature=0.2,
        system=SYS_PROMPT,
        messages=[{'role': 'user', 'content': prompt}],
    )

    return response.content[0].text


def gpt4o_mini(prompt: str) -> str:
    '''
    Call the GPT-4o-mini
    '''
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.2,
        max_tokens=1024,
        messages=[{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content


def judge_llm(prompt: str) -> str:
    '''
    Judge the response of a language model
    '''
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini', temperature=0.2, max_tokens=1024, messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content


def evaluation(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    for llm_name in df['llm'].unique():
        for video_name in df['video'].unique():
            for k in df['k'].unique():

                df_ = df[(df['llm'] == llm_name) & (df['video'] == video_name) & (df['k'] == k)]
                candidates = df_['llm_response'].tolist()
                references = df_['true_answer'].tolist()

                rouge1, rougeL = metrics.rouge_score(references, candidates)
                meteor = metrics.meteor_score(references, candidates)

                ref_embeds = embedder.encode(references)
                cand_embeds = embedder.encode(candidates)
                similarity = util.pytorch_cos_sim(ref_embeds, cand_embeds).diag().mean().item()

                relevance, correctness = metrics.llm_judge_score(
                    judge_llm, LLM_JUDGE_PROMPT, references, candidates, df_['context'].tolist()
                )

                results.append(
                    {
                        'llm': llm_name,
                        'video': video_name,
                        'k': k,
                        'rouge1': rouge1,
                        'rougeL': rougeL,
                        'meteor': meteor,
                        'embedding_similarity': similarity,
                        'llm_relevance': relevance,
                        'llm_correctness': correctness,
                    }
                )
                # low level
                logger.info(f'{llm_name} - {video_name} - {k} - ROUGE1: {rouge1} - ROUGEL: {rougeL} - METEOR: {meteor}')
                # high level
                logger.info(
                    f'{llm_name} - {video_name} - {k} - Embed. sim. {similarity} - Relevance: {relevance} - Correctness: {correctness}'
                )

    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the RAG model')
    parser.add_argument('--results_path', default=None, help='The results to evaluate')
    args = parser.parse_args()

    if args.results_path is not None:
        if not os.path.exists(args.results_path):
            logger.error('The results path does not exist')
            exit(1)

        df = pd.read_csv(args.results_path)

        eval_results = evaluation(df)
        eval_results.to_csv('results/rag_evaluation_metrics.csv', index=False)
        print(eval_results.to_string(index=False))

        logger.success('Evaluation is finished')
        exit(0)

    results = []

    for llm in ['haiku', 'gpt4o_mini']:
        for video_name in ['SamAltman_interview']:  # VIDEOS.keys(): -> to save money and time
            df = pd.read_csv(f'../data/{video_name}_gt.csv')

            # create rag
            documents = []
            for chunk_id in df['chunk_id'].unique():
                doc = df[df['chunk_id'] == chunk_id].iloc[0]['chunk']
                documents.append({'id': chunk_id, 'text': doc})

            index = minsearch.Index(text_fields=['text'], keyword_fields=['id'])
            index.fit(documents)

            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'Evaluating {video_name} with {llm}'):
                question = row['question']
                retrieved_docs = index.search(question, num_results=3)

                # with top 1 response
                prompt = LLM_PROMPT.format(question=question, transcript=retrieved_docs[0]['text'])

                if llm == 'haiku':
                    response = haiku(prompt)
                elif llm == 'gpt4o_mini':
                    response = gpt4o_mini(prompt)

                results.append(
                    {
                        'llm': llm,
                        'video': video_name,
                        'question': question,
                        'llm_response': response,
                        'true_answer': row['answer'],
                        'context': retrieved_docs[0]['text'],
                        'k': 1,
                    }
                )

                # with top 3 responses
                transcripts = '\n'.join([r['text'] for r in retrieved_docs])
                prompt = LLM_PROMPT.format(question=question, transcript=transcripts)

                if llm == 'haiku':
                    response = haiku(prompt)
                elif llm == 'gpt4o_mini':
                    response = gpt4o_mini(prompt)

                results.append(
                    {
                        'llm': llm,
                        'video': video_name,
                        'question': question,
                        'llm_response': response,
                        'true_answer': row['answer'],
                        'context': transcripts,
                        'k': 3,
                    }
                )

                if len(results) % 10 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv('results/rag_evaluate_results.csv', index=False)

    # save results
    df = pd.DataFrame(results)
    df.to_csv('results/rag_evaluate_results.csv', index=False)

    logger.success('Results are saved')
