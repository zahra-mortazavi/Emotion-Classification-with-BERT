# Emotion Classification with BERT

This project implements an emotion classification model using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. The model is trained on the GoEmotions dataset to classify text into one of six emotions: **anger**, **fear**, **joy**, **love**, **sadness**, and **surprise**. It supports multilingual text, leveraging the `bert-base-multilingual-uncased` model.

The project includes a Jupyter Notebook for training and evaluation, an inference script for predictions, and utilities for logging and visualization.

## Features
- Fine-tuning BERT for emotion classification.
- Multilingual support (e.g., English, Persian).
- Training with early stopping based on validation accuracy.
- Visualization of training loss and confusion matrices.
- Saving best and final models.
- Inference script for easy predictions on new text.
- Sample predictions and downloadable artifacts (models, logs, plots).

## Prerequisites
- Python 3.8+
- GPU recommended for training (e.g., Google Colab with T4 GPU).
- Libraries: `transformers`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `datasets`.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd emotion-classification-bert
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (If no `requirements.txt` is provided, install manually:)
   ```
   pip install transformers torch scikit-learn pandas matplotlib seaborn tqdm datasets
   ```

## Usage

### Training the Model
1. Open the Jupyter Notebook: `Emotion_Classification_with_BERT.ipynb`.
2. Run all cells to:
   - Load and preprocess the GoEmotions dataset.
   - Fine-tune the BERT model for 3 epochs (configurable).
   - Evaluate on validation set.
   - Save models, logs, confusion matrices, and training metrics.

   Key configurations (in the notebook):
   - Batch size: 16 (train), 32 (validation/test).
   - Learning rate: 2e-5.
   - Max sequence length: 128.
   - Early stopping patience: 2 epochs.

   Outputs:
   - `best_model/`: Directory with the best model based on validation accuracy.
   - `final_model/`: Directory with the model after the last epoch.
   - `training_metrics.csv`: Epoch-wise loss and accuracy.
   - `training_loss.png`: Plot of training loss over epochs.
   - `confusion_matrix_epoch_X.png`: Confusion matrices per epoch.
   - Zipped models: `best_model.zip` and `final_model.zip` for download.

### Inference
Use the provided `inference.py` script for predictions on new text.

Example command:
```
python inference.py --text "I am very happy today" --model_dir "best_model"
```

Options:
- `--text`: Input text (required).
- `--model_dir`: Path to the model directory (default: `best_model`).
- `--max_len`: Max sequence length (default: 128).
- `--batch_size`: Batch size for predictions (default: 32).

For batch predictions in code:
```python
from inference import EmotionClassifier

classifier = EmotionClassifier("best_model")
texts = ["I feel very happy today", "من خیلی عصبانیم"]
results = classifier.predict(texts)
print(results)
# Output: [{'text': '...', 'label': 'joy', 'confidence': 0.9995}, ...]
```

## Results
- Training on GoEmotions dataset (simplified to 6 emotions).
- Validation accuracy after 3 epochs: ~92.7%.
- Sample predictions:
  - "I am so happy today" → joy (99.95% confidence).
  - "This is very sad news" → sadness (98.73% confidence).
  - "من امروز خیلی خوشحالم" (Persian for "I am very happy today") → joy (88.6% confidence).

Training metrics example:
| Epoch | Train Loss | Val Accuracy |
|-------|------------|--------------|
| 1     | 0.743     | 0.9085      |
| 2     | 0.207     | 0.9250      |
| 3     | 0.139     | 0.9270      |

## Limitations
- Trained on a subset of emotions; may not generalize to all nuances.
- Performance on low-resource languages may vary.
- Requires GPU for efficient training/inference.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


For questions or contributions, open an issue or pull request.
