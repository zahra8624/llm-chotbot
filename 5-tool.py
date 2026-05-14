from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

from datetime import datetime, timezone

get_current_datetime_schema = {
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description": "Get the current date and time in a specified timezone with configurable output format. Returns datetime string, timezone, and unix timestamp.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone_str": {
                    "type": "string",
                    "enum": ["UTC", "EST", "PST"],
                    "description": "The timezone to return the current time in. Supported values: UTC (Coordinated Universal Time), EST (Eastern Standard Time, UTC-5), PST (Pacific Standard Time, UTC-8)",
                },
                "format": {
                    "type": "string",
                    "enum": ["iso", "readable"],
                    "description": "Output format for the datetime string. 'iso' returns ISO 8601 format (e.g., 2024-01-15T14:30:00), 'readable' returns human-friendly format (e.g., January 15, 2024 at 2:30 PM)",
                },
            },
            "required": [],
        },
    },
}


def get_current_datetime(timezone_str="UTC", format="iso"):
    """
    Get the current date and time in specified timezone and format
    """
    # Get current time in UTC
    now_utc = datetime.now(timezone.utc)

    # For simplicity, we'll handle just a few timezones
    # In production, use pytz library
    if timezone_str == "UTC":
        current_time = now_utc
    elif timezone_str == "EST":
        current_time = now_utc.replace(hour=(now_utc.hour - 5) % 24)
    elif timezone_str == "PST":
        current_time = now_utc.replace(hour=(now_utc.hour - 8) % 24)
    else:
        current_time = now_utc

    # Format the output
    if format == "iso":
        return {
            "datetime": current_time.isoformat(),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    elif format == "readable":
        return {
            "datetime": current_time.strftime("%B %d, %Y at %I:%M %p"),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    else:
        return {
            "datetime": str(current_time),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }


def process_tool_calls(client, messages, response):
    """Process tool calls and return the final response"""

    choice = response.choices[0]

    if choice.finish_reason == "tool_calls":
        # Add assistant's response to messages
        messages.append(choice.message)

        # Process each tool call
        for tool_call in choice.message.tool_calls:
            print(f"\n🔧 Tool call detected: {tool_call.function.name}")
            args = json.loads(tool_call.function.arguments)
            print(f"Parameters: {args}")

            # Execute the tool
            if tool_call.function.name == "get_current_datetime":
                result = get_current_datetime(**args)
                print(f"Result: {result}\n")

                # Append tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        # Send tool results back to the model
        followup_response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages,
            tools=[get_current_datetime_schema],
        )

        # If there are more tool calls, recursively process them
        if followup_response.choices[0].finish_reason == "tool_calls":
            return process_tool_calls(client, messages, followup_response)

        return followup_response

    return response


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)
    messages = []

    print("🕐 Interactive Time Zone Chatbot")
    print("Ask me about the current time in different places!")
    print("Supported timezones: UTC, EST (Eastern), PST (Pacific/California)")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check if user wants to exit
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n👋 Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        print(f"\n{'='*60}")

        # Get initial response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages,
            tools=[get_current_datetime_schema],
        )

        # Process any tool calls and get final response
        final_response = process_tool_calls(client, messages, response)

        # Display the final answer
        final_choice = final_response.choices[0]
        print(f"{'='*60}")
        print(f"Assistant: {final_choice.message.content}")
        print(f"{'='*60}\n")

        # Add assistant's final response to conversation history
        if final_choice.finish_reason != "tool_calls":
            messages.append(
                {
                    "role": "assistant",
                    "content": final_choice.message.content,
                }
            )


if __name__ == "__main__":
    main()
