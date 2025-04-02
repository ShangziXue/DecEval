import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
import os
import random

class RSPPairDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, neg_samples=3):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data) * (self.neg_samples + 1)

    def __getitem__(self, idx):
        if idx % (self.neg_samples + 1) == 0:
            # positive
            q, r, s = self.data[idx // (self.neg_samples + 1)]
            return self._process_pair(q, r, s)
        else:
            # negative
            q, r_pos, s_pos = self.data[(idx - 1) // (self.neg_samples + 1)]
            r_neg = random.choice([item[1] for item in self.data if item[0] != q])
            return self._process_pair(q, r_neg, 0)

    def _process_pair(self, q, r, s):
        encoding = self.tokenizer(
            f"[CLS] {q} [SEP] {r} [SEP]",
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor([s])
        }

class RSP(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(RSP, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        h_cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        h_combined = torch.cat([h_cls[:self.bert.config.hidden_size], 
                               h_cls[self.bert.config.hidden_size:]], dim=1)
        alpha = self.mlp(h_combined)
        return alpha.squeeze()

def train_rsp(model, train_loader, val_loader, epochs=3, lr=2e-5, save_path='./checkpoints'):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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
        
        val_loss = evaluate_rsp(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_path, 'best_model.pth')
            print(f"New best model saved with val loss: {val_loss:.4f}")
    
    save_model(model, save_path, 'final_model.pth')
    return model

def evaluate_rsp(model, val_loader, criterion, device):
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

def save_model(model, save_path, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': model.bert.config.model_name_or_path,
            'hidden_size': model.bert.config.hidden_size
        }
    }, os.path.join(save_path, filename))

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = RSP(model_name=config['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_relevance(model, question, reasoning, tokenizer, device, threshold=0.5):
    model.eval()
    encoding = tokenizer(
        f"[CLS] {question} [SEP] {reasoning} [SEP]",
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        alpha = model(encoding['input_ids'], encoding['attention_mask'])
    
    return alpha.item() >= threshold

if __name__ == "__main__":
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SAVE_PATH = './rsp_checkpoints'
    THRESHOLD = 0.7
    
    # example data
    data = [
        ("How are A and B compared?", "Compare their prices", 1),
        ("What's the combination of X and Y?", "Calculate total cost", 1),
        ("How are A and B compared?", "Calculate total cost", 0),
        ("What's the combination of X and Y?", "Compare historical trends", 0)
    ]
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = RSPPairDataset(data, tokenizer)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    model = RSP(model_name=MODEL_NAME)
    
    trained_model = train_rsp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path=SAVE_PATH
    )
    
    test_question = "Evaluate blockchain technology impact"
    test_reasoning = "Analyze energy consumption patterns"
    
    loaded_model = load_model(os.path.join(SAVE_PATH, 'best_model.pth'), device='cuda')
    
    score = predict_relevance(
        model=loaded_model,
        question=test_question,
        reasoning=test_reasoning,
        tokenizer=tokenizer,
        device='cuda',
        threshold=THRESHOLD
    )
    
    print(f"Relevance Score: {score:.4f}")
    print("Reasoning step accepted" if score >= THRESHOLD else "Reasoning step rejected")