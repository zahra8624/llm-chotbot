from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                # "content": "perform a web search for the latest news on renewable energy advancement i 2024 and summarize the top three findings.  ",
                "content": "Whats the  weather in Isfahan right now ?",
            }
        ],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
    )


if __name__ == "__main__":
    main()
