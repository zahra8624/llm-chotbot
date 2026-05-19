from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime

load_dotenv()

product_review_schema = {
    "type": "function",
    "function": {
        "name": "extract_product_review",
        "description": """Extract and enrich product review information from messy, unstructured text. 
        This tool handles real-world reviews with informal language, emojis, slang, and various rating formats.
        It normalizes data, infers missing information, and enriches the review with additional insights.""",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title or headline of the review. If not explicitly stated, infer a concise title (3-8 words) that captures the essence of the review.",
                },
                "review_text": {
                    "type": "string",
                    "description": "The cleaned and normalized full text of the review, without informal abbreviations or excessive emojis.",
                },
                "rating": {
                    "type": "integer",
                    "description": "Rating on a 1-5 scale. Convert from other formats: '10/10' or '5 stars' → 5, '3.5/5' → 4, '8/10' → 4, etc. If no explicit rating, infer from sentiment.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "author": {
                    "type": "string",
                    "description": "Username or name of the reviewer. If not found, use 'Anonymous'.",
                },
                "product_name": {
                    "type": "string",
                    "description": "The specific product being reviewed, extracted from the text.",
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "Overall sentiment of the review based on the language and tone used.",
                },
                "would_recommend": {
                    "type": "boolean",
                    "description": "Whether the reviewer would recommend this product, inferred from the review content.",
                },
                "key_features_mentioned": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific product features or aspects mentioned in the review.",
                },
                "price_mentioned": {
                    "type": "number",
                    "description": "Price mentioned in the review, if any. Extract the numeric value only.",
                },
                "pros": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of positive aspects or advantages mentioned.",
                },
                "cons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of negative aspects or drawbacks mentioned.",
                },
            },
            "required": [
                "title",
                "review_text",
                "rating",
                "author",
                "product_name",
                "sentiment",
                "would_recommend",
            ],
        },
    },
}


def extract_product_review(
    title,
    review_text,
    rating,
    author,
    product_name,
    sentiment,
    would_recommend,
    key_features_mentioned=None,
    price_mentioned=None,
    pros=None,
    cons=None,
):
    structured_review = {
        "title": title,
        "review_text": review_text,
        "rating": rating,
        "author": author,
        "product_name": product_name,
        "sentiment": sentiment,
        "would_recommend": would_recommend,
        "key_features_mentioned": key_features_mentioned or [],
        "price_mentioned": price_mentioned,
        "pros": pros or [],
        "cons": cons or [],
        "extracted_at": datetime.now().isoformat(),
        "processed": True,
    }
    return structured_review


def process_tool_calls(client, messages, response):
    choice = response.choices[0]

    if choice.finish_reason != "tool_calls":
        return response

    messages.append(choice.message)

    for tool_call in choice.message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"\n🔧 Tool call detected: {name}")
        print(f"📊 Extracted Data:")
        print(json.dumps(args, indent=2))

        if name == "extract_product_review":
            result = extract_product_review(**args)
            print(f"\n✅ Review structured and enriched successfully!")
            print(f"\n💾 Structured Output:")
            print(json.dumps(result, indent=2))

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

    followup = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=messages,
        tools=[product_review_schema],
    )

    if followup.choices[0].finish_reason == "tool_calls":
        return process_tool_calls(client, messages, followup)

    return followup


def run_example(client, label, challenge, review_text):
    print(f"\n\n📦 {label}")
    print("=" * 80)
    print(f"💡 Challenge: {challenge}\n")
    print(f"Raw Review Text:\n{review_text}\n")

    messages = [
        {
            "role": "user",
            "content": f"Extract and enrich the product review information from this text:\n\n{review_text}",
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=messages,
        tools=[product_review_schema],
        tool_choice={"type": "function", "function": {"name": "extract_product_review"}},
    )

    process_tool_calls(client, messages, response)
    print(f"\n{'='*80}\n")


def main():
    client = OpenAI(base_url="https://api.gapgpt.app/v1", timeout=30.0)

    print("🔍 ENHANCED Product Review Extraction with Data Enrichment")
    print("=" * 80)
    print("This demo shows structured data extraction with tool calling:")
    print("  ✓ Handles messy, informal, real-world text")
    print("  ✓ Normalizes ratings from different formats (10/10, stars, etc.)")
    print("  ✓ Infers missing information intelligently")
    print("  ✓ Enriches data with sentiment, features, pros/cons")
    print("  ✓ Extracts and validates structured JSON")
    print("=" * 80)

    run_example(
        client,
        label="EXAMPLE 1: Messy Informal Review (Real-World Data)",
        challenge="Informal language, emojis, non-standard rating format",
        review_text="""
        omg these headphones r AMAZING!!! 🎧🔥 i paid like 200 bucks last week 
        and theyre totally worth it. battery life is insane - i think 30hrs? 
        maybe more idk. noise canceling is chef's kiss 👨‍🍳💋 sound quality is 
        crispy af. only downside is they're kinda heavy after a while but nbd.
        10/10 would buy again
        - tech_guy_92
        """,
    )

    run_example(
        client,
        label="EXAMPLE 2: Missing Rating - Sentiment-Based Inference",
        challenge="No explicit rating, must infer from negative sentiment",
        review_text="""
        Subject: Total waste of money - BrewMaster 3000
        
        I bought this coffee maker hoping for café-quality drinks at home. What a joke!
        The coffee is mediocre at best. Machine sounds like a jet engine taking off.
        Takes FOREVER to brew one cup. Water tank is tiny - refill constantly.
        At $150 this is highway robbery. Save your money and go to Starbucks instead.
        
        Posted by: CoffeeLover_Jane
        """,
    )

    run_example(
        client,
        label="EXAMPLE 3: Mixed Sentiment Review",
        challenge="Both positive and negative points, partial star rating",
        review_text="""
        Galaxy Watch Ultra - decent but not perfect
        
        Got this smartwatch 2 weeks ago for $399. GPS tracking is super accurate 
        for my runs and the heart rate monitor seems legit. Battery easily lasts 
        2 days which is nice. Screen is bright and crisp.
        
        However... the UI is clunky and confusing. Lots of features I'll never use.
        Notifications are buggy sometimes. For the price, I expected more polish.
        
        3.5 stars - good hardware, software needs work
        
        ~ FitnessFanatic23
        """,
    )

    run_example(
        client,
        label="EXAMPLE 4: Ultra-Short Review (Edge Case)",
        challenge="Minimal text, must infer context and details",
        review_text="""
        AirPods Max - overpriced. Sound is good but not $549 good. ⭐⭐⭐
        """,
    )

    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\n🎯 What This Demo Showed:")
    print("\n1. 🧹 DATA NORMALIZATION")
    print("\n2. 🧠 INTELLIGENT INFERENCE")
    print("\n3. 📊 DATA ENRICHMENT")
    print("\n4. ✅ STRUCTURED VALIDATION")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()