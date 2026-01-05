# Image detector: Summary and VQA

The Image Summary module provides advanced image analysis capabilities using the Qwen2.5-VL multimodal model. Qwen2.5-VL is a multimodal large language model capable of understanding and generating content from both images and videos. With its help, AMMMICO supports tasks such as image/video summarization and image/video visual question answering, where the model answers users' questions about the context of a media file. It combines functionality from the `model.py` and `prompt_builder.py` modules to offer comprehensive image understanding.

## Core Components

### MultimodalSummaryModel (`model.py`)

The underlying model wrapper that handles Qwen2.5-VL model loading and inference:

- **Model Selection**: 
  - CUDA: `Qwen/Qwen2.5-VL-7B-Instruct` (default for GPU)
  - CPU: `Qwen/Qwen2.5-VL-3B-Instruct` (default for CPU)
- **Automatic Device Detection**: Auto-detects CUDA availability and falls back to CPU
- **Quantization**: Automatic 4-bit quantization for CUDA devices using BitsAndBytesConfig
- **Memory Management**: Resource cleanup methods for long-running processes
- **Model Components**: Provides processor, tokenizer, and model objects

### PromptBuilder (`prompt_builder.py`)

Modular prompt construction system for multi-level analysis:

- **Processing Levels**: Frame, Clip, and Video level prompts
- **Task Types**: Summary generation, VQA, or combined tasks
- **Audio Integration**: Prompts that incorporate audio transcription when available
- **Structured Output**: Ensures consistent, well-formatted model outputs

## Key Features

- **Image Captioning**: Generate concise or detailed captions for images
- **Visual Question Answering (VQA)**: Answer custom questions about image content
- **Batch Processing**: Process multiple images efficiently with configurable batch sizes
- **Flexible Input**: Supports file paths, PIL Images, or sequences of images
- **Analysis Types**:
  - `summary`: Generate image captions only
  - `questions`: Answer questions only
  - `summary_and_questions`: Both caption and Q&A (default)
- **Concise Mode**: Option to generate shorter, more focused summaries and answers
- **Question Chunking**: Automatically processes questions in batches (default: 8 per batch)
- **Error Handling**: Robust error handling with retry logic for CUDA operations

## Usage

```python
from ammico.image_summary import ImageSummaryDetector
from ammico.model import MultimodalSummaryModel

# Initialize model
model = MultimodalSummaryModel(device="cpu")

# Create detector
detector = ImageSummaryDetector(summary_model=model, subdict={})

# Analyze single image
results = detector.analyse_image(
    entry={"filename": "image.jpg"},
    analysis_type="summary_and_questions",
    list_of_questions=["What is in this image?", "Are there people?"],
    is_concise_summary=True,
    is_concise_answer=True
)

# Batch processing
detector.subdict = image_dict
results = detector.analyse_images_from_dict(
    analysis_type="summary",
    keys_batch_size=16
)
```

## Configuration

- **Max Questions**: Default 32 questions per image (configurable)
- **Batch Size**: Default 16 images per batch (configurable)
- **Token Limits**: 
  - Concise summary: 64 tokens
  - Detailed summary: 256 tokens
  - Concise answers: 64 tokens
  - Detailed answers: 128 tokens

## Output

Returns dictionaries with:
- `caption`: Generated image caption (if summary requested)
- `vqa`: List of answers corresponding to questions (if questions requested)