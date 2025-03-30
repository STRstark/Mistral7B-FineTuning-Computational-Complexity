import os
import json
import random

DATA_PATH = "/home/moahamdreza/Desktop/Ai/JSONFIles"
output = []

SPECIALIZED_PROMPTS = [
    "Explain a concept involving {keywords} in computational complexity.",
    "This is a technical discussion about {keywords} in theoretical computer science.",
    "Read and understand this section about {keywords} from a research paper.",
    "Scientific context on {keywords} and their role in graph theory.",
    "Understanding the computational aspects of {keywords}."
]

for file in os.listdir(DATA_PATH):
    if file.endswith(".json"):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk in data:
                chunk_text = chunk.get("chunk_text", "").strip()
                metadata = chunk.get("metadata", {})
                keywords = metadata.get("keywords", [])
                selected_keywords = ", ".join(keywords[:3]) if keywords else "a specific problem"

                if chunk_text:
                    prompt_template = random.choice(SPECIALIZED_PROMPTS)
                    prompt = prompt_template.format(keywords=selected_keywords)
                    output.append({
                        "prompt": prompt,
                        "response": chunk_text
                    })

with open("fine_tune_data.json", "w", encoding="utf-8") as out_file:
    json.dump(output, out_file, indent=2, ensure_ascii=False)
