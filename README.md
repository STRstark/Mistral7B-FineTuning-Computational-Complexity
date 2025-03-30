
# 📘 Fine-Tuning a Small LLM on Scientific Text Chunks (Computational Complexity)

This guide documents the full pipeline for preparing and fine-tuning a small language model like **Mistral 7B** on custom scientific texts (e.g., research papers from arXiv in the field of computation complexity), using a consumer GPU like RTX 3060.


## 🧾 Overview of the Pipeline

1. **Extract text from scientific PDFs**
2. **Chunk and preprocess text**
3. **Generate structured JSON with metadata and keywords**
4. **Embed text chunks into a vector database (ChromaDB)**
5. **Convert data to a format suitable for fine-tuning**
6. **Fine-tune a causal LLM using LoRA on a local GPU**


## 1.  Extract and Chunk Text from PDF

Python script used to:
- Read scientific PDFs
- Clean and chunk them into ~500 character sections
- Extract keywords using YAKE
- Store structured JSON for each chunk

**Key script:** `pdf_to_json.py`

```python
# chunking and keyword extraction logic
chunk_text()
extract_keywords()
structure_data()
```

Each JSON looks like this:

```json
{
  "id": "uuid",
  "title": "Doc Title",
  "chunk_text": "Scientific content...",
  "metadata": {
    "author": "Author",
    "keywords": ["Subset", "Sum", "NP-complete"]
  }
}
```

## 2. 🧪 Convert Chunks to Fine-tuning Format

Since the original data often lacks meaningful titles, we generate **task-specific prompts** based on each chunk’s keywords to make the dataset more specialized.

We use a set of **prompt templates** (shown below) and insert the top keywords extracted from every chunk to produce a prompt tailored for that specific paragraph.

```python
SPECIALIZED_PROMPTS = [
    "Explain a concept involving {keywords} in computational complexity.",
    "This is a technical discussion about {keywords} in theoretical computer science.",
    "Read and understand this section about {keywords} from a research paper.",
    "Scientific context on {keywords} and their role in graph theory.",
    "Understanding the computational aspects of {keywords}."
]
```
**Prompt template examples:**

- `Explain a concept involving {keywords} in computational complexity.`
- `Scientific context on {keywords} and their role in graph theory.`

**Output format:**

```json
{
  "prompt": "Explain a concept involving Subset, Sum, NP-complete in computational complexity.",
  "response": "The Subset Sum problem, which asks whether a given set of n integers..."
}
```

**Script used:** `convert_json_for_finetune.py`

---

## 4. 🏗️ Fine-Tuning the Model (with LoRA)

### Requirements

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

(Optional: `xformers` for better performance)

---

### Authenticate with Hugging Face

1. Create a token: https://huggingface.co/settings/tokens
2. Log in:

```bash
huggingface-cli login
```

---

### Fine-Tuning Script (`train_lora.py`)

Using `mistralai/Mistral-7B-v0.1` (or any LoRA-compatible causal model):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
import json

# Load dataset
with open("fine_tune_data.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# Load model and tokenizer
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Tokenize
def tokenize(example):
    return tokenizer(
        f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize)

# Training setup
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

---

## 5. ✅ Inference (After Fine-Tuning)

```python
model.eval()
inputs = tokenizer("Explain the subset sum problem.", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 💡 Tips for Low-GPU (RTX 3060)

- Use LoRA or QLoRA instead of full fine-tuning
- Keep batch size = 1 and use `gradient_accumulation_steps`
- Use `fp16=True` and `max_length=512` or less
- Monitor GPU memory with `nvidia-smi`

---

## ✅ Future Extensions

- Replace prompt generation with actual instructions
- Add evaluation script for generated outputs
- Experiment with QLoRA + `trl` for chat-style tuning

---

**Made with ❤️ and a single RTX 3060 :)**

---

## 📄 PDF to Structured JSON Conversion

This script processes raw scientific PDFs and converts them into structured JSON format, ready for both embedding and fine-tuning.

**Key script:** `PdfToJSON.py`

### 🔧 Responsibilities of this script:

- Open and read PDF pages using PyMuPDF (`fitz`)
- Clean and normalize raw text (`clean_text`)
- Split long text into chunks (~500 characters) by sentence (`chunk_text`)
- Extract top keywords from each chunk using YAKE (`extract_keywords`)
- Wrap each chunk with metadata into a consistent structured JSON format

### 🔍 Code Highlights:

```python
chunk_text(text, chunk_size=500)
# Splits text into manageable chunks based on sentence boundaries.

extract_keywords(text, max_keywords=10)
# Uses YAKE to extract top keywords from each chunk.

structure_data(chunks, metadata)
# Combines chunked text with metadata and extracted keywords.
```

Each final JSON chunk looks like this:

```json
{
  "id": "uuid",
  "title": "",
  "chunk_text": "Cleaned and segmented text here...",
  "metadata": {
    "author": "Extracted author or empty string",
    "keywords": ["keyword1", "keyword2", ...]
  }
}
```

The output is saved in `JSONFiles/` with a unique file name per input PDF.