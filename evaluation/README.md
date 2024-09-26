

# Dataset Generation

Using gpt4o-mini



# Retrieval Evaluation



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


While TF-IDF may not be the top-performing retrieval method in terms of accuracy, it offers a compelling balance between performance and computational efficiency that makes it the preferred choice for our system. The decrease in retrieval performance compared to more complex methods like hybrid search is not substantial enough to outweigh its significant benefits. TF-IDF's notable speed advantage is crucial for systems requiring real-time or near-real-time responses, and its efficiency makes it highly scalable, suitable for large datasets or high-traffic applications. Moreover, TF-IDF is simpler to implement and maintain compared to more complex algorithms, reducing development time and ongoing operational costs. Its lower computational requirements lead to cost savings in terms of hardware and energy consumption. In essence, choosing TF-IDF represents a strategic decision that prioritizes a pragmatic balance between performance, speed, and resource utilization, allowing for a responsive, scalable, and cost-effective retrieval system without significantly compromising on the quality of results.


# RAG Evalutation

Best Retriever

RAG: 1 or 3 top retrievals
LLMs: `gpt4o-mini`, `claude-haiku`

Metrics: `Sentence transformers cos-similarity`, `Rouge`, `meteor`, `LLM Judge`

## SamAltman_interview

| LLM         | k | Rouge1  | RougeL  | Meteor | Embedding Similarity | LLM Relevance | LLM Correctness |
|-------------|---|---------|---------|--------|----------------------|---------------|-----------------|
| haiku       | 1 | 0.104   | 0.095   | 0.233  | 0.471                | 4.222         | 3.884           |
| haiku       | 3 | 0.098   | 0.090   | 0.227  | 0.478                | 4.474         | 4.267           |
| gpt4o_mini  | 1 |**0.188**|**0.169**| 0.330  | 0.488                | 4.219         | 3.938           |
| gpt4o_mini  | 3 | 0.180   | 0.164   |**0.334**| **0.512**           | **4.535**     | **4.395**       |


As table shows, our default choice LLM is `gpt4o-mini` and we will provide top 3 retrieved documents

# Conclusion
based on above best choice for retrieval algorithm, LLM and prompt:
-
-
-
