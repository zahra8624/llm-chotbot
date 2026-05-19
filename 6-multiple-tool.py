from dotenv import load_dotenv
from openai import OpenAI
import json
import random
from datetime import datetime, timezone

load_dotenv()

get_current_datetime_schema = {
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description": "Get the current date and time in a specified timezone with configurable output format.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone_str": {
                    "type": "string",
                    "enum": ["UTC", "EST", "PST"],
                    "description": "UTC, EST (UTC-5), PST (UTC-8)",
                },
                "format": {
                    "type": "string",
                    "enum": ["iso", "readable"],
                    "description": "'iso' = 2024-01-15T14:30:00 | 'readable' = January 15 2024 at 2:30 PM",
                },
            },
            "required": [],
        },
    },
}

roll_dice_schema = {
    "type": "function",
    "function": {
        "name": "roll_dice",
        "description": "Roll one or more dice with a specified number of sides.",
        "parameters": {
            "type": "object",
            "properties": {
                "num_dice": {
                    "type": "integer",
                    "description": "Number of dice to roll (1-10)",
                    "minimum": 1,
                    "maximum": 10,
                },
                "sides": {
                    "type": "integer",
                    "enum": [4, 6, 8, 10, 12, 20, 100],
                    "description": "Sides on each die",
                },
            },
            "required": [],
        },
    },
}


def get_current_datetime(timezone_str="UTC", format="iso"):
    now_utc = datetime.now(timezone.utc)

    if timezone_str == "EST":
        current_time = now_utc.replace(hour=(now_utc.hour - 5) % 24)
    elif timezone_str == "PST":
        current_time = now_utc.replace(hour=(now_utc.hour - 8) % 24)
    else:
        current_time = now_utc

    if format == "readable":
        return {
            "datetime": current_time.strftime("%B %d, %Y at %I:%M %p"),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    return {
        "datetime": current_time.isoformat(),
        "timezone": timezone_str,
        "unix_timestamp": int(current_time.timestamp()),
    }


def roll_dice(num_dice=1, sides=6):
    rolls = [random.randint(1, sides) for _ in range(num_dice)]
    return {
        "dice_count": num_dice,
        "sides": sides,
        "rolls": rolls,
        "total": sum(rolls),
        "average": round(sum(rolls) / len(rolls), 2),
        "max_possible": num_dice * sides,
    }


TOOL_MAP = {
    "get_current_datetime": get_current_datetime,
    "roll_dice": roll_dice,
}


def process_tool_calls(client, messages, response, tools):
    choice = response.choices[0]

    if choice.finish_reason != "tool_calls":
        return response

    messages.append(choice.message)

    for tool_call in choice.message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"\n🔧 Tool call: {name}")
        print(f"   Args: {args}")

        fn = TOOL_MAP.get(name)
        result = fn(**args) if fn else {"error": f"Unknown tool: {name}"}
        print(f"   Result: {result}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    followup = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )

    if followup.choices[0].finish_reason == "tool_calls":
        return process_tool_calls(client, messages, followup, tools)

    return followup


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)
    tools = [get_current_datetime_schema, roll_dice_schema]
    messages = []

    print("Assistant with Multiple Tools")
    print("=" * 50)
    print("Available tools:")
    print("• Current time in UTC, EST, or PST")
    print("• Roll dice (d4, d6, d8, d10, d12, d20, d100)")
    print("\nExamples:")
    print("- 'What time is it in California and roll 2 six-sided dice'")
    print("- 'Roll 3d20 and tell me the time in EST'")
    print("\nType 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        print(f"\n{'='*60}")

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages,
            tools=tools,
        )

        final = process_tool_calls(client, messages, response, tools)

        answer = final.choices[0].message.content
        print(f"{'='*60}")
        print(f"Assistant: {answer}")
        print(f"{'='*60}\n")

        if final.choices[0].finish_reason != "tool_calls":
            messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
