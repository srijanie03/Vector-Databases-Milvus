#Import libraries
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from FlagEmbedding import BGEM3FlagModel


#Setup
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "bge_m3_doc_collection"  # Milvus collection name
EMBEDDING_MODEL = "BAAI/bge-m3"


test_sentences = "What is BGE M3?"
model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
test_embedding = model.encode([test_sentences])['dense_vecs'][0]

print(f'{test_embedding[:20]} ...')
dimension = len(test_embedding)
print(f'\nDimensions of `{EMBEDDING_MODEL}` embedding model is: {dimension}')


# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)


# Set scheme with 3 fields: id (int), text (string), and embedding (float array).
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
]
schema = CollectionSchema(fields, "Here is description of this collection.")


# Create a collection with above schema.
doc_collection = Collection(COLLECTION_NAME, schema)

# Create an index for the collection.
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}
doc_collection.create_index("embedding", index)



#Read file
with open('./m3_paper.txt', 'r') as f:
    lines = f.readlines()

embeddings = model.encode(lines)['dense_vecs']
entities = [
    list(range(len(lines))),  # field id (primary key) 
    lines,  # field text
    embeddings,  #field embedding
]
insert_result = doc_collection.insert(entities)

# In Milvus, it's a best practice to call flush() after all vectors are inserted,
# so that a more efficient index is built for the just inserted vectors.
doc_collection.flush()




# Load the collection into memory for searching
doc_collection.load()


def semantic_search(query, top_k=3):
    vectors_to_search = model.encode([query])['dense_vecs']
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }
    result = doc_collection.search(vectors_to_search, "embedding", search_params, limit=top_k, output_fields=["text"])
    return result[0]


question = 'How many working languages does the M3-Embedding model support?'
match_results = semantic_search(question, top_k=3)


#Write the output to file
file1 = open('integration_output_cosine.txt', 'w')
from datetime import datetime 
current_date_time = datetime.now() 

file1.write(current_date_time.strftime("%Y-%m-%d %H:%M:%S"))
file1.write("\n")


for match in match_results:
    print(f"distance = {match.distance:.2f}\n{match.entity.text}")
    file1.write(str(match.distance))
    file1.write("\n")
    file1.write(match.entity.text)
    file1.write("\n")
file1.close()