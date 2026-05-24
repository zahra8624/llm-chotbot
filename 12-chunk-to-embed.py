from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
import lancedb

load_dotenv()


def save_to_lancedb(embedded_chunks, table_name="faq_embeddings", db_path="./lancedb"):
    print(f"\n💾 Saving embeddings to LanceDB...")
    print(f"   Database path: {db_path}")
    print(f"   Table name: {table_name}")

    db = lancedb.connect(db_path)

    data = [
        {
            "id": f"chunk_{chunk['chunk_id']}",
            "text": chunk["text"],
            "vector": chunk["embedding"],
            "source": chunk["metadata"].get("source", ""),
        }
        for chunk in embedded_chunks
    ]

    if table_name in db.table_names():
        db.drop_table(table_name)

    table = db.create_table(table_name, data=data)

    print(f"   ✅ Saved {len(embedded_chunks)} embeddings to LanceDB")
    print(f"   Table size: {len(table)} items")

    return table


def load_document(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\n📄 Loading file: {file_path.name}")
    print(f"   File type: {file_path.suffix}")
    print(f"   File size: {file_path.stat().st_size / 1024:.2f} KB")

    if file_path.suffix == ".txt":
        loader = TextLoader(str(file_path))
    elif file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} document(s)")

    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    print(f"\n✂️  Chunking documents...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Chunk overlap: {chunk_overlap} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"   ✅ Created {len(chunks)} chunks")

    if chunks:
        print(f"\n📝 Sample chunk (first 200 chars):")
        print(f"   {chunks[0].page_content[:200]}...")

    return chunks


def generate_embedding(client, text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def embed_chunks(client, chunks):
    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")

    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...", end="\r")

        embedding = generate_embedding(client, chunk.page_content)

        embedded_chunks.append(
            {
                "chunk_id": i,
                "text": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata,
            }
        )

    print(f"\n   ✅ Generated {len(embedded_chunks)} embeddings")
    print(f"   Embedding dimension: {len(embedded_chunks[0]['embedding'])}")

    return embedded_chunks


def generate_answer_with_context(query, retrieved_docs, client):
    print(f"\n🤖 Generating answer with GPT-4o...")

    context = "\n\n".join(
        [f"Context {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Use ONLY the information from the provided context to answer the question
- If the context doesn't contain enough information to answer the question, say so
- Be concise but thorough in your answer
- Cite which context section you're using when relevant"""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )

    answer = response.choices[0].message.content

    print(f"   ✅ Answer generated")
    print(f"\n" + "=" * 70)
    print("💬 GPT-4o ANSWER:")
    print("=" * 70)
    print(answer)
    print("=" * 70)

    return answer


def search_and_answer(table, query_text, client, n_results=3):
    print(f"\n🔍 Searching for: '{query_text}'")
    print(f"   Retrieving top {n_results} relevant chunks...")

    query_embedding = generate_embedding(client, query_text)

    results = table.search(query_embedding).limit(n_results).to_list()

    print(f"\n📋 Retrieved Context:")
    print("=" * 70)

    retrieved_docs = []
    for i, result in enumerate(results):
        print(f"\n🔹 Context {i+1} (Distance: {result['_distance']:.4f})")
        print(f"   {result['text'][:200]}...")
        print("-" * 70)
        retrieved_docs.append(result["text"])

    answer = generate_answer_with_context(query_text, retrieved_docs, client)

    return {"query": query_text, "answer": answer}


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)

    print("=" * 70)
    print("📚 Document Processing & Embedding Pipeline with RAG")
    print("=" * 70)

    db_path = "./lancedb"
    table_name = "faq_embeddings"

    db = lancedb.connect(db_path)
    existing_tables = db.table_names()

    if table_name in existing_tables:
        print(f"\n✅ Found existing table: '{table_name}'")
        table = db.open_table(table_name)
        print(f"   Table contains {len(table)} embeddings")
        print(f"   Skipping document loading and embedding process...")

        print("\n🔹 SEARCH & ANSWER DEMO (RAG)")
        print("-" * 70)

        query = input(
            "\nEnter a question about the document (or press Enter to skip): "
        ).strip()

        if query:
            result = search_and_answer(table, query, client, n_results=3)
            print(f"\n💾 Full response stored in result variable")
        else:
            print("   Skipping search demo")

        print("\n💡 To re-process documents, delete the ./lancedb folder")
        return None

    print("\n🔹 STEP 1: Upload & Load Document")
    print("-" * 70)

    file_path = input("Enter file path (or press Enter for demo text): ").strip()

    if not file_path:
        demo_file = Path("FAQ.txt")
        file_path = str(demo_file)
        print(f"📝 Using demo file: {file_path}")

    try:
        documents = load_document(file_path)

        print("\n🔹 STEP 2: Chunk Documents")
        print("-" * 70)
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

        print("\n🔹 STEP 3: Generate Embeddings")
        print("-" * 70)
        embedded_chunks = embed_chunks(client, chunks)

        print("\n🔹 STEP 4: Sample Embeddings")
        print("-" * 70)
        num_samples = min(3, len(embedded_chunks))

        for i in range(num_samples):
            chunk_data = embedded_chunks[i]
            print(f"\n📊 Chunk {i+1} Embedding:")
            print(f"   Text preview: {chunk_data['text'][:100]}...")
            print(
                f"   Embedding vector (first 10 values): {chunk_data['embedding'][:10]}"
            )
            print(
                f"   Embedding vector (last 10 values): {chunk_data['embedding'][-10:]}"
            )
            print(f"   Total dimensions: {len(chunk_data['embedding'])}")

        print("\n🔹 STEP 5: Save to LanceDB")
        print("-" * 70)
        table = save_to_lancedb(embedded_chunks, table_name=table_name, db_path=db_path)

        print("\n🔹 STEP 6: Search & Answer Demo (RAG)")
        print("-" * 70)
        print("\n💡 This demonstrates Retrieval Augmented Generation (RAG):")
        print("   1. Your question is converted to an embedding")
        print("   2. Most relevant document chunks are retrieved")
        print("   3. GPT-4o uses those chunks to generate an accurate answer")

        query = input(
            "\nEnter a question about the document (or press Enter to skip): "
        ).strip()

        if query:
            result = search_and_answer(table, query, client, n_results=3)
            print(f"\n💾 Full response stored in result variable")
            print(f"   - result['query']: Your question")
            print(f"   - result['answer']: GPT-4o's answer")
        else:
            print("   Skipping RAG demo")

        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📊 Summary:")
        print(f"   • Original documents: {len(documents)}")
        print(f"   • Total chunks: {len(chunks)}")
        print(f"   • Embeddings generated: {len(embedded_chunks)}")
        print(f"   • Embedding dimension: {len(embedded_chunks[0]['embedding'])}")
        print(f"   • Saved to LanceDB: {len(table)} items")
        print(f"   • Database location: {db_path}")
        print(f"\n💡 Your RAG system is ready!")
        print(f"   - Documents are embedded and stored")
        print(f"   - Next run will use existing embeddings (faster!)")
        print(f"   - GPT-4o answers based on YOUR documents")
        print("=" * 70)

        return embedded_chunks

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    main()
