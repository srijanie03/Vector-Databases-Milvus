import milvus
import pymilvus
from pymilvus import connections
connections.connect(host="localhost",port=19530)

from pymilvus import FieldSchema, CollectionSchema, DataType

# number of dimensions in the embedding model
# for sentence-transformers/all-MiniLM-L12-v2, that's 384
DIMENSION = 384

# id and embedding are required to define
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]


# "enable_dynamic_field" lets us insert data with any metadata fields
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)



from pymilvus import Collection

# define the collection name and pass the schema
collection = Collection(name="example_name", schema=schema)

index_params = {
    "index_type": "IVF_FLAT", # one of 11 Milvus indexes, IVF is the most intuitive
    "metric_type": "L2", # L2, Cosine, or IP
    "params": {"nlist": 4}, # how many "centroids" do you want for IVF?
}


# pass the field to index on and the parameters to index with
collection.create_index(field_name="embedding", index_params=index_params)
# load the collection into memory
collection.load()

from sentence_transformers import SentenceTransformer

# a popular 384 dimension vector embedding model
transformer = SentenceTransformer('all-MiniLM-L12-v2')


# Input file
with open("./Seattle.txt", "r") as f:
    x = f.read() # read the entire file in as a string

# Split on the number of sentences for simplicity
sentences = x.split(".")


# Hold embeddings and sentences together
milvus_input = []
for sentence in sentences:
    entry = {}
    vector_embedding = transformer.encode(sentence)
    entry["embedding"] = vector_embedding
    entry["sentence"] = sentence
    milvus_input.append(entry)


# milvus expects a list of dicts 
collection.insert(milvus_input)
collection.flush()


#Reading data

query = "the tallest point in Seattle"
q_embedding = transformer.encode(query)


res = collection.search(
    data=[q_embedding],  # Embeded search value
    anns_field="embedding",  # Search across embeddings
    param={"metric_type": "L2",
            "params": {"nprobe": 2}},
    limit = 3,  # Limit to top_k results per search
    output_fields=["sentence"]  # Include title field in result
)


#Write the output to file

file1 = open('myfile.txt', 'w')

from datetime import datetime 
current_date_time = datetime.now() 

file1.write(current_date_time.strftime("%Y-%m-%d %H:%M:%S"))
file1.write("\n")

for i, hits in enumerate(res):
    for hit in hits:
        print(hit.entity.get("sentence"))
        print(hit.entity.id)
        file1.write(hit.entity.get("sentence"))
file1.close()
 
