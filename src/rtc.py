import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
import os

class RTCDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


class RTC(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        super(RTC, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.predictor = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        v_qc = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        logits = self.predictor(v_qc)
        return logits, v_qc  

def save_checkpoint(model, tokenizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': model.encoder.config.model_name_or_path,
            'num_labels': model.predictor.out_features
        }
    }, path)
    tokenizer.save_pretrained(path)

def load_checkpoint(path, model_class, tokenizer_class, model_name):
    checkpoint = torch.load(path)
    config = checkpoint['config']
    
    tokenizer = tokenizer_class.from_pretrained(config['model_name'])
    
    model = model_class(model_name=config['model_name'], num_labels=config['num_labels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5, save_path='./checkpoints'):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, f"{save_path}/best_model.pth")
            print(f"New best model saved with val loss: {val_loss:.4f}")
    
    save_checkpoint(model, tokenizer, f"{save_path}/final_model.pth")
    return model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def predict(model, tokenizer, text, device, threshold=0.5):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        logits = model(**encoding)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    predictions = {
        'scores': probs,
        'prediction': None
    }
    
    if probs.max() >= threshold:
        predicted_idx = probs.argmax()
        predictions['prediction'] = predicted_idx
    
    return predictions

if __name__ == "__main__":
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 4  
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = './rtc_checkpoints'
    
    train_texts = ["How are A and B compared?", "What's the combination of X and Y?"]
    train_labels = [[1,0,0,0], [0,1,0,0]]  
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = RTCDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = RTC(model_name=MODEL_NAME, num_labels=NUM_LABELS)
    
    trained_model, trained_tokenizer = train_model(
        model, 
        train_loader, 
        train_loader, 
        epochs=EPOCHS,
        save_path=MODEL_SAVE_PATH
    )
    
    loaded_model, loaded_tokenizer = load_checkpoint(
        f"{MODEL_SAVE_PATH}/best_model.pth",
        RTC,
        BertTokenizer,
        MODEL_NAME
    )
    
    test_text = "Compare the features of product A and product B"
    prediction = predict(loaded_model, loaded_tokenizer, test_text, device='cuda')
    print(f"Predicted class probabilities: {prediction['scores']}")
    print(f"Predicted class index: {prediction['prediction']}")