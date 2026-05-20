from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)

    print("\n🌐 OpenAI Web Search Demo")
    print("=" * 60)
    print("Ask a question that requires current information!\n")

    question = input("Your question: ").strip()
    if not question:
        question = "What's the weather in NYC?"
        print(f"Using default question: {question}")

    print("\n⏳ Searching the web...\n")

    response = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview"}],
        input=question,
    )

    print("=" * 60)
    print("📝 Answer:\n")

    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    print(content.text)

                    if hasattr(content, "annotations") and content.annotations:
                        print("\n📚 Sources:")
                        for ann in content.annotations:
                            if ann.type == "url_citation":
                                print(f"  • {ann.title}")
                                print(f"    {ann.url}\n")

    print("\n" + "=" * 60)
    print("💰 Usage:")
    print(f"  Input tokens:  {response.usage.input_tokens}")
    print(f"  Output tokens: {response.usage.output_tokens}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
