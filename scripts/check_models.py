import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN is missing from .env")
    exit()

# قائمة بأشهر وأقوى النماذج المجانية الداعمة للمحادثة
models_to_test = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-1.1-7b-it"
]

client = InferenceClient(api_key=HF_TOKEN)

print("=" * 60)
print("TESTING FREE CHAT MODELS ON HUGGING FACE")
print("=" * 60)

working_models = []

for model in models_to_test:
    print(f"Testing: {model.ljust(40)}", end="")
    try:
        # محاولة إرسال طلب Chat Completion بسيط جداً
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("✅ WORKING")
        working_models.append(model)
    except Exception as e:
        # طباعة الخطأ باختصار
        error_msg = str(e).split('\n')[0]
        print(f"❌ FAILED ({error_msg})")

print("\n" + "=" * 60)
if working_models:
    print("🎯 THE FOLLOWING MODELS ARE READY TO USE IN YOUR .env:")
    for m in working_models:
        print(f"HF_EXTRACTION_MODEL={m}")
else:
    print("⚠️ All tested models failed. Hugging Face free tier might be overloaded.")
print("=" * 60)