from config import GROQ_API_KEY, LLM_PROVIDER

def get_llm(lite=False):
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        model = "llama-3.1-8b-instant" if lite else "llama-3.3-70b-versatile"
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=model,
            temperature=0.2
        )
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from config import ANTHROPIC_API_KEY
        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model="claude-sonnet-4-6",
            temperature=0.2
        )