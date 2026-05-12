from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

def add_user_message(messages, content):
    messages.append({"role":"user", "content":content})

def add_assistant_message(messages, content):
    messages.append({"role":"assistant", "content":content}) 

def chat(messages):
    client= OpenAI(base_url="https://api.gapgpt.app/v1")
    response=client.chat.completions.create(
        model="gpt-5-nano",
        max_tokens=3000,
        messages=messages
    )
    # print("*************************************************")
    # print(response.choices)
    # print("*************************************************")
    return response.choices[0].message.content

def display_messages(messages):
    """Display all messages begin to sent to the LLM"""
    print("\n" + "=" *60)
    print("MESSAGES BEGIN SENT TO LLM:")
    print("=" *60)
    for i, msg in enumerate(messages,1):
        role= msg["role"].upper()
        content=msg["content"]
        print(f"\n[Message {i} - {role}]")
        print(f"{content}")
    print("=" * 60 + "\n")    

def main():
    messages=[]

    print("Simple console chatbot")
    print("Type 'quit', 'exit' or 'bye' to end the conversation.\n ")

    while True:
        #get user input
        user_input=input("You: ").strip()
        if user_input.lower() in ["quit","exit", "buy"]:
            print("\nGoodBuy!")
            break
        if not user_input:
            continue
        add_user_message(messages,user_input)
        display_messages(messages)

        print("Thinking...\n")
        response=chat(messages)
        print(f"Assistant: {response} \n")
        add_assistant_message(messages,response)
    

if __name__ == "__main__":
    main()
