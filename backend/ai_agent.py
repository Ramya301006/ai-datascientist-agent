"""
AI Agent — powered by Claude (Anthropic).
Replaces the old hardcoded HuggingFace stub.
"""
import os
import anthropic

# ── Put your key in a .env file or set the env var:
#    ANTHROPIC_API_KEY=sk-ant-...
# ── The client reads it automatically from the environment.
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", "YOUR_KEY_HERE")
)

SYSTEM_PROMPT = """You are an expert AI Data Scientist assistant.
You help users understand their datasets, explain machine learning concepts,
interpret model results, suggest analysis steps, and answer data science questions.
Be concise, accurate, and practical. Use bullet points for lists.
When given dataset context, refer to it in your answer."""


def ask_claude(question: str, dataset_context: str = "") -> str:
    """
    Send a question to Claude and return the answer as a string.
    Optionally pass dataset_context (column names, stats, etc.)
    so Claude can give dataset-aware answers.
    """
    user_content = question
    if dataset_context:
        user_content = f"Dataset context:\n{dataset_context}\n\nQuestion: {question}"

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_content}
            ]
        )
        return message.content[0].text

    except anthropic.AuthenticationError:
        return "❌ Invalid API key. Set ANTHROPIC_API_KEY in your .env file."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit hit. Please wait a moment and try again."
    except Exception as e:
        return f"❌ Claude API error: {str(e)}"


def ask_claude_about_dataset(question: str, df) -> str:
    """
    Ask Claude a question with full dataset awareness.
    Passes shape, columns, dtypes, and sample stats as context.
    """
    try:
        context_lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            f"Dtypes:\n{df.dtypes.to_string()}",
            f"Missing values:\n{df.isnull().sum().to_string()}",
            f"Sample statistics:\n{df.describe(include='all').to_string()}",
        ]
        context = "\n".join(context_lines)
    except Exception:
        context = ""

    return ask_claude(question, dataset_context=context)