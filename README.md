# Review Sentiment Classifier - PyTorch + FastAPI

Fine-tuned **DistilBERT** for binary sentiment classification of customer reviews,
with a custom **PyTorch** training loop and a **FastAPI** inference service.

## Results
- Training data: 4,000 Yelp reviews | Test data: 1,000 unseen reviews
- **Test accuracy: 91.3% | F1: 0.905** after 2 epochs

## How it works
1. Tokenisation with a custom PyTorch `Dataset` and `DataLoader`
2. Fine-tuning DistilBERT with a manual training loop (forward pass, loss, backpropagation, AdamW)
3. Evaluation with accuracy and F1 on a held-out test set
4. Serving the model via FastAPI (`/health`, `/predict`), containerised with Docker

## Run the API
    pip install -r requirements.txt
    uvicorn app:app --reload
    # POST /predict  {"text": "Great hotel, friendly staff!"}

Model weights are not stored in this repo (size); train them with the notebook.
