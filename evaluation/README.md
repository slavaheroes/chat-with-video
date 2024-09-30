# Evaluation of Pipeline

`requirements.txt` is different. Hence you need to run `pip install -r requirements.txt` one more time to be able to run files in this folder.
Also, you need to run `Elastic search` using Docker:
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```


> This readme file was created using Claude Sonnet 3.5

## Table of Contents
1. [Dataset Generation](#dataset-generation)
2. [Retrieval Evaluation](#retrieval-evaluation)
3. [RAG Evaluation](#rag-evaluation)
4. [Conclusion](#conclusion)

## Dataset Generation

The `generate_datasets.py` file is responsible for creating the dataset used in our evaluation. Here's a brief overview of its functionality:

1. It processes YouTube videos specified in the `VIDEOS` dictionary.
    - [Sam Altman interview to Lex Fridman podcast](https://www.youtube.com/watch?v=jvqFAi7vkBc)
    - [A documentary about Stalin repressions](https://www.youtube.com/watch?v=oo1WouI38rQ)
    - [Pavel Durov interview to Tucker Carlson](https://www.youtube.com/watch?v=1Ut6RouSs0w)
2. For each video, it:
   - Extracts metadata (author, title, description)
   - Loads and processes the transcript
   - Splits the transcript into chunks
   - Uses `gpt-4o-mini` to generate question-answer pairs for each chunk
3. The generated data is saved as CSV files, one for each video, containing chunk IDs, chunk content, generated questions, and answers.

This approach allows us to create a comprehensive dataset for evaluating our retrieval and RAG systems.

### To reproduce:
``` bash
python generate_datasets.py
```

## Retrieval Evaluation

We evaluated four different retrieval methods:

1. **Hybrid search using Elastic Search**: Combines keyword-based search with vector similarity search for improved results.
2. **Default Elastic Search settings**: Uses out-of-the-box Elastic Search configuration for text retrieval.
3. **TF-IDF**: Implemented as in the course, using term frequency-inverse document frequency for ranking documents.
4. **Custom VDB**: A custom vector database implemented using Sentence Transformers for embedding-based retrieval.

Results for each video:

### SamAltman_interview

| Method       | Hit Rate | MRR     | Latency (s)    |
|--------------|----------|---------|------------|
| hybrid_es    | **0.798** | **0.564**| 32.647  |
| default_es   | 0.781 | 0.547| 4.473   |
| if_idf       | 0.760 | 0.519| **1.927**   |
| custom_vdb   |  0.664 | 0.397| 17.950 |

### vDud_Kolyma

| Method       | Hit Rate | MRR     | Latency (s)   |
|--------------|----------|---------|------------|
| hybrid_es    | **0.864** | **0.631**| 34.962  |
| default_es   |  0.846 | 0.612| 5.427   |
| if_idf       | 0.843| 0.592| **2.254**   |
| custom_vdb   | 0.793 | 0.507| 22.974 |

### PavelDurov_interview

| Method       |  Hit Rate | MRR     | Latency (s)    |
|--------------|----------|---------|------------|
| hybrid_es    |  **0.877** | **0.657**| 11.665  |
| default_es   |  0.858 | 0.636| 1.927  |
| if_idf       |  0.848 | 0.617| **0.745**  |
| custom_vdb   |  0.817 | 0.532| 7.705  |

Although TF-IDF isn't the most accurate retrieval method, we've chosen it for our system because it strikes a good balance between performance and efficiency. While it's slightly less accurate than more complex methods like hybrid search, the difference isn't big enough to ignore its major advantages. TF-IDF is much faster, which is crucial for quick responses, and it works well with large amounts of data or high-traffic systems. It's also easier to set up and maintain than more complicated algorithms, which saves time and money. TF-IDF doesn't need as much computing power, which means we can use less expensive hardware and save on energy costs. In short, we picked TF-IDF because it gives us a good mix of speed, efficiency, and quality results, while keeping our system responsive and cost-effective.

### To reproduce:
``` bash
python evaluate_retrieval.py
```

## RAG Evaluation

In this ablation study, we compared two different Language Models (`gpt4o-mini` and `claude-haiku`) with both top 1 and top 3 retrieved documents. The results demonstrate that providing more context (top 3 documents) generally leads to better performance across various metrics. This is particularly evident in the improved Embedding Similarity, LLM Relevance, and LLM Correctness scores when using top 3 documents.

`gpt4-o-mini` is a better-performing language model than `claude-haiku`.

> `Note:`: `gpt4-o-mini` serves as the LLM Judge to evaluate the Relevance and Correctness. To save cost and time, the evaluation was done only on one video.

### SamAltman_interview

| LLM         | k | Rouge1  | RougeL  | Meteor | Embedding Similarity | LLM Relevance | LLM Correctness |
|-------------|---|---------|---------|--------|----------------------|---------------|-----------------|
| haiku       | 1 | 0.104   | 0.095   | 0.233  | 0.471                | 4.222         | 3.884           |
| haiku       | 3 | 0.098   | 0.090   | 0.227  | 0.478                | 4.474         | 4.267           |
| gpt4o_mini  | 1 |**0.188**|**0.169**| 0.330  | 0.488                | 4.219         | 3.938           |
| gpt4o_mini  | 3 | 0.180   | 0.164   |**0.334**| **0.512**           | **4.535**     | **4.395**       |

### To reproduce:
``` bash
python evaluate_rag.py
python evaluate_rag.py --results_path results/rag_evaluate_results.csv
```

## Conclusion

Based on the comprehensive evaluation results presented above, we have made the following decisions for our pipeline:

1. **Retrieval Algorithm**: We select TF-IDF as our retrieval method. Despite not being the top performer in terms of accuracy, it offers the best balance between performance and computational efficiency. Its speed and scalability make it ideal for real-time applications, and the marginal decrease in accuracy is outweighed by its benefits in terms of simplicity, maintainability, and resource utilization.

2. **Language Model**: We choose GPT4-o-mini as our language model. It consistently outperformed claude-haiku across various metrics, demonstrating superior performance in generating relevant and correct responses.

3. **Context Provision**: We will provide the top 3 retrieved documents to the language model. The results clearly show that increasing the context from 1 to 3 documents leads to improvements across all metrics, particularly in terms of relevance and correctness as judged by the LLM.

These choices aim to create a robust, efficient, and accurate RAG system that balances performance with practical considerations of speed and resource usage.