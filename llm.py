from config import (GROQ_API_KEY, LLM_PROVIDER, GROQ_MODEL, GROQ_MODEL_LITE,
                    GROQ_REASONING_EFFORT)


def get_llm(lite=False):
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        # Model IDs come from config.py (single source of truth), not hardcoded.
        kwargs = dict(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_LITE if lite else GROQ_MODEL,
            temperature=0.2,
        )
        effort = (GROQ_REASONING_EFFORT or "").strip().lower()
        if effort and effort != "none":
            kwargs["reasoning_effort"] = effort
        return ChatGroq(**kwargs)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model=ANTHROPIC_MODEL,
            temperature=0.2
        )
