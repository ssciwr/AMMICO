import os

import ammico

# Define your data path
data_path = "."  # the current directory


# Configure the externally hosted vision-language model (OpenAI-compatible endpoint).
# For a self-hosted vLLM server, AMMICO_API_KEY is the value passed to `vllm serve --api-key`.

# vllm on same machine
USE_VLLM = False
if USE_VLLM:
    os.environ["AMMICO_API_BASE_URL"] = "http://localhost:8000/v1"
    # set model and key

# ollama on same machine
USE_OLLAMA = False
if USE_OLLAMA:
    os.environ["AMMICO_API_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["AMMICO_API_KEY"] = "ollama"
    os.environ["AMMICO_MODEL_ID"] = "qwen2.5vl:3b"
    os.environ["AMMICO_PRIVACY_ACK"] = "True"  # using ollama

# locally hosted qwen model
USE_HGSMATHCOMP = True
if USE_HGSMATHCOMP:
    os.environ["AMMICO_API_BASE_URL"] = "http://************:9000/v1"
    os.environ["AMMICO_API_KEY"] = "*************"
    os.environ["AMMICO_MODEL_ID"] = "Qwen/Qwen3.8-27B"

# --- OpenRouter with a GPT model -------------------------------------------
# Set USE_OPENROUTER = True to send the requests to OpenRouter instead of hgsmathcomp.
# OpenRouter exposes an OpenAI-compatible endpoint; the API key is read from the
# OPENROUTER_API_KEY environment variable (export OPENROUTER_API_KEY=sk-or-...).
# Note: the image data is sent to a third party (OpenRouter / OpenAI).
USE_OPENROUTER = False
if USE_OPENROUTER:
    os.environ["AMMICO_API_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["AMMICO_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    # any vision-capable model from https://openrouter.ai/models, prefixed with the provider
    os.environ["AMMICO_MODEL_ID"] = "openai/gpt-4o-mini"
    os.environ["AMMICO_PRIVACY_ACK"] = "True"  # acknowledge sending data externally
# ---------------------------------------------------------------------------


USE_VIDEO = False
if USE_VIDEO:
    os.environ["AMMICO_AUDIO_BASE_URL"] = (
        "http://ammico.ammico.svc:8081/v1"  # Whisper endpoint
    )
    os.environ["AMMICO_AUDIO_API_KEY"] = "sk-speaches"
    os.environ["AMMICO_AUDIO_MODEL_ID"] = (
        "Systran/faster-whisper-large-v3"  # or whisper-1
    )


class NoThinkingInferenceModel(ammico.InferenceModel):
    """InferenceModel that disables the reasoning ("thinking") phase of Qwen3 models.

    Otherwise the reasoning consumes the whole max_tokens budget and the
    returned content (the caption) is empty.
    """

    def chat(self, messages, max_new_tokens=256, n=1):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            n=n,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return [(choice.message.content or "").strip() for choice in response.choices]


if __name__ == "__main__":
    # Find files
    image_dict = ammico.find_files(path=data_path, limit=10)
    video_dict = ammico.find_videos(path=data_path, limit=10)

    # reads the AMMICO_API_* environment variables; the Qwen3-specific
    # "no thinking" switch is not needed (and not understood) by GPT models
    model = NoThinkingInferenceModel() if USE_HGSMATHCOMP else ammico.InferenceModel()
    if USE_VIDEO:
        audio_model = ammico.AudioTranscriptionModel()  # optionally language="en"

    image_summary_vqa = ammico.ImageSummaryDetector(
        summary_model=model, subdict=image_dict
    )
    is_concise_summary = True
    summary = image_summary_vqa.analyse_images_from_dict(
        analysis_type="summary", is_concise_summary=is_concise_summary
    )

    for key in summary:
        image_path = summary[key]["filename"]
        print(f"Summary: {summary[key]['caption']}", image_path)
