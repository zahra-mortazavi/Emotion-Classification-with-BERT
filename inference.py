import torch
import argparse
from typing import List, Dict
from transformers import BertTokenizer, BertForSequenceClassification


class EmotionClassifier:
    def __init__(self, model_path: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Example label mapping (modify based on your dataset)
        self.id2label = {
            0: "anger",
            1: "fear",
            2: "joy",
            3: "love",
            4: "sadness",
            5: "surprise"
        }

    def predict(self, texts: List[str]) -> List[Dict]:

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        predictions = torch.argmax(probs, dim=1)

        results = []

        for text, pred, prob in zip(texts, predictions, probs):

            label = self.id2label[pred.item()]
            confidence = prob[pred].item()

            results.append({
                "text": text,
                "label": label,
                "confidence": round(confidence, 4)
            })

        return results


def main():

    parser = argparse.ArgumentParser(description="Emotion Classification Inference")

    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model",
        help="Path to trained model"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text for prediction"
    )

    args = parser.parse_args()

    classifier = EmotionClassifier(args.model_path)

    results = classifier.predict([args.text])

    for r in results:
        print("\nText:", r["text"])
        print("Predicted Emotion:", r["label"])
        print("Confidence:", r["confidence"])


if __name__ == "__main__":
    main()