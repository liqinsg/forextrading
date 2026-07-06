import os
from openai import OpenAI
from google import genai
from utils.schemas import TradeSignal

# ==========================================
# 0. FEATURE FLAGS
# ==========================================
use_qwen = os.getenv("QWEN_API_KEY") is not None
use_deepseek = os.getenv("DEEPSEEK_API_KEY") is not None

# ==========================================
# 1. INITIALIZE CLIENTS (ONLY IF AVAILABLE)
# ==========================================
gemini_client = genai.Client()

qwen_client = None
if use_qwen:
    qwen_client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

deepseek_client = None
if use_deepseek:
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )


# ==========================================
# 2. MODEL CALLERS
# ==========================================
def get_gemini_decision(prompt: str) -> TradeSignal:
    """Queries Gemini 2.5 Flash using strict structural schemas."""
    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TradeSignal,
            'temperature': 0.1,
        },
    )
    return TradeSignal.model_validate_json(response.text)


def get_qwen_decision(prompt: str) -> TradeSignal:
    """Queries Qwen-Max only if enabled."""
    if not use_qwen or qwen_client is None:
        raise RuntimeError("Qwen is disabled or API key not set")

    response = qwen_client.chat.completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    return TradeSignal.model_validate_json(
        response.choices[0].message.content
    )


def get_deepseek_decision(prompt: str) -> TradeSignal:
    """Queries DeepSeek only if enabled."""
    if not use_deepseek or deepseek_client is None:
        raise RuntimeError("DeepSeek is disabled or API key not set")

    response = deepseek_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    return TradeSignal.model_validate_json(
        response.choices[0].message.content
    )


# ==========================================
# 3. OPTIONAL: UNIFIED ROUTER (BEST PRACTICE)
# ==========================================
def get_trade_decision(prompt: str, provider: str = "gemini") -> TradeSignal:
    """
    Unified entry point for model selection.
    provider: "gemini" | "qwen" | "deepseek"
    """

    if provider == "gemini":
        return get_gemini_decision(prompt)

    elif provider == "qwen":
        if use_qwen:
            return get_qwen_decision(prompt)
        raise RuntimeError("Qwen not available")

    elif provider == "deepseek":
        if use_deepseek:
            return get_deepseek_decision(prompt)
        raise RuntimeError("DeepSeek not available")

    else:
        raise ValueError(f"Unknown provider: {provider}")