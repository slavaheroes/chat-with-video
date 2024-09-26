import json

import evaluate
from loguru import logger


def rouge_score(references, candidates):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=candidates, references=references)
    return results['rouge1'], results['rougeL']


def meteor_score(references, candidates):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=candidates, references=references)
    return results["meteor"]


def llm_judge_score(llm_func, prompt_template, references, candidates, context):

    relevance_score = 0.0
    correctness_score = 0.0
    count = 0

    for gt, candidate, ctx in zip(references, candidates, context):
        prompt = prompt_template.format(true_answer=gt, llm_answer=candidate, context=ctx)
        response = llm_func(prompt)
        try:
            response = json.loads(response)
            relevance_score += response["relevance_score"]
            correctness_score += response["correctness_score"]
            count += 1
        except Exception as e:
            logger.info(f"Error: {e}")
            logger.info(f"Response: {response}")

    return relevance_score / count, correctness_score / count


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + (1 / (rank + 1))

    return total_score / len(relevance_total)
