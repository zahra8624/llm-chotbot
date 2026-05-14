from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():

    client = OpenAI(base_url="https://api.gapgpt.app/v1")
    print("=" * 85 + "\n")
    print(
        "Generate a CloudFormation  template in JSON format that provision an EC2 instance"
    )
    print("\n" + "=" * 85)
    print("Assistant: ", end="", flush=True)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a CloudFormation  template in JSON format that provision an EC2 instance. Return ONLY valid JSON, no explanation."
                ),
            }
        ],
        response_format={"type": "json_object"},
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
