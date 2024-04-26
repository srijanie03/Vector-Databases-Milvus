# Vector-Databases-Milvus
This repo contains notebooks of my experiments/learnings in Milvus. 

I wrote about vector databases in my article here:

https://medium.com/towards-data-science/deep-dive-into-vector-databases-by-hand-e9ab71f54f80

and seeing Milvus in action while being able to draw correspondence to the theory was super!

**Codes**

I tried the intergration of BGE-M3 model with Milvus. For each sentence in the paper, we use BAAI/bge-m3 model to convert the text string into 1024 dimension vector embedding, and store each embedding in Milvus.The input text was the M3 paper itself.

**Steps**:
1. Each sentence in the input file is first converted into a vector embedding of dimension 1024 and stored into Milvus.
2. A 'collection' is created with the fields 'id', 'text' and 'embedding'.
3. The next step is indexing where a shorter version of the embedding is produced for faster retrieval. Milvus does the heavy work with the parameters specified.
4. Finally the query text is converted to vector embedding and compared with the vector embeddings obtained from the text.
5. I try both cosine similarity metric and the L2 metric, both provide the same result. Note: Smaller the value, closer the vectors and hence a better match. Howeever, for cosine similarity the values are reversed - higher means better. Reason being cos(0°)=1, and thus a value closer to 1 will mean a smaller angular difference between the vectors and hence a closer match.
