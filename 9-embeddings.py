from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_embedding(
    client, text, model="text-embedding-3-large", input_type="query"
):
    response = client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)

    text = "The quick brown fox jumps over the lazy dog."
    embedding = generate_embedding(client, text)

    print(f"Generated Embedding (first 5 values): {embedding[:5]}")
    print(f"Embedding dimensions: {len(embedding)}")


if __name__ == "__main__":
    main()
