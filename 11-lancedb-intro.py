import lancedb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


db = lancedb.connect("./lancedb")


documents = [
    ("1", "These oranges are delicious."),
    ("2", "Some apples are sweet."),
    ("3", "Citrus fruits are great."),
]

data = [
    {"id": id, "text": text, "vector": get_embedding(text)} for id, text in documents
]

table = db.create_table("my_test_collection", data=data, mode="overwrite")


query_embedding = get_embedding("tell me about citrus fruits")

results = table.search(query_embedding).limit(2).to_list()

print("Query Results:")
for r in results:
    print(f"  ID: {r['id']}, Text: '{r['text']}', Distance: {r['_distance']:.4f}")
