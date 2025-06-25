# nodes/inference_node.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class InferenceNode:
    def __init__(self):
        model_path = "./sentiment-model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def __call__(self, state):
        # ✅ Fix: support dataclass or dict input
        input_text = state.text if hasattr(state, "text") else state["text"]

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=1)

        label = "positive" if predicted.item() == 1 else "negative"
        return {
            "text": input_text,
            "label": label,
            "confidence": round(confidence.item() * 100, 2),
        }
