
# **Fine-Tuning a Small LLM on Scientific Text Chunks (Computational complexity)**

This guide documents the full pipeline for preparing and fine-tuning a small language model like **Mistral 7B** on custom scientific texts (e.g., research papers from arXiv in the field of computation complexity), using a consumer GPU like RTX 3060.


## Overview of the Pipeline

1. **Extract text from scientific PDFs**
2. **Chunk and preprocess text**
3. **Generate structured JSON with metadata and keywords**
4. **Convert data to a format suitable for fine-tuning**
5. **Fine-tune a causal LLM using LoRA on a local GPU**


## **1.  Extract and Chunk Text from PDF**

Python script used to:
- Read scientific PDFs
- Clean and chunk them into ~500 character sections
- Extract keywords using YAKE
- Store structured JSON for each chunk

**Key script:** `PdfToJSON.py`

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

**Output format example**

```json
{
  "prompt": "Explain a concept involving Subset, Sum, NP-complete in computational complexity.",
  "response": "The Subset Sum problem, which asks whether a given set of n integers..."
}
```

**Script used:** `FineTuningData.py`

**Output File :** `fine_tune_data.json`

---

## 3. Fine-Tuning the Model (with LoRA)

Using `mistralai/Mistral-7B-v0.1` (or any LoRA-compatible causal model):

- ### Load dataset

```python
with open("fine_tune_data.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
```
- ### Load model and tokenizer
```python
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
```
- ### Apply LoRA
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
```
- ### Tokenize it!
```python
def tokenize(example):
    return tokenizer(
        f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize)
```
- ### Traine the Model
```python
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
### Fine-Tuning Script (`Tunung.py`)

## 4. Inference (After Fine-Tuning)

```python
model.eval()
inputs = tokenizer("Explain the subset sum problem.", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## 💡 Tips for Low-GPU (RTX 3060)

- Use LoRA or QLoRA instead of full fine-tuning
- Keep batch size = 1 and use `gradient_accumulation_steps`
- Use `fp16=True` and `max_length=512` or less
- Monitor GPU memory with `nvidia-smi`


## **Made with ❤️ and a single RTX 3060 :)**
### **To be honest it never ran because of the lack of prossesing power :(**

