from rtc import RTC 
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class NSGDataset(Dataset):
    def __init__(self, prev_subqs, next_subqs, tokenizer, max_length=512):
        self.prev_subqs = prev_subqs
        self.next_subqs = next_subqs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prev_subqs)

    def __getitem__(self, idx):
        prev_subq = str(self.prev_subqs[idx])
        next_subq = str(self.next_subqs[idx])
        
        encoding = self.tokenizer(
            prev_subq,
            next_subq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten() 
        }

class PrefixInjector(nn.Module):
    def __init__(self, input_dim, prefix_dim=128):
        super(PrefixInjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, prefix_dim),
            nn.ReLU(),
            nn.Linear(prefix_dim, prefix_dim*2)  
        )

    def forward(self, v_qcpred):
        return self.mlp(v_qcpred)

class NSGWithRTC(nn.Module):
    def __init__(self, rtc_model, bart_model_name='facebook/bart-large'):
        super().__init__()
        self.rtc = rtc_model  
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.prefix_injector = PrefixInjector(self.bart.config.hidden_size)
        self.eos_token_id = self.bart.config.eos_token_id
        self.max_steps = 10

    def generate_step(self, 
                      current_context, 
                      rtc_encoder, 
                      tokenizer, 
                      device):
        inputs = tokenizer(
            current_context,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            v_qc, _ = self.rtc(inputs.input_ids, inputs.attention_mask)
        
        outputs = self.bart.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=100,
            num_beams=5,
            early_stopping=True,
            prefix_allowed_tokens_fn=lambda _1, _2: [tokenizer.encode('[EOQ]')[0]]
        )
        
        new_subq = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_subq = new_subq.split('[EOQ]')[0].strip()
        
        if new_subq:
            current_context += f" {new_subq}"
        return new_subq, v_qc

    def forward(self, 
                initial_question, 
                rtc_encoder, 
                tokenizer, 
                device):
        sub_questions = []
        current_context = initial_question
        
        for step in range(self.max_steps):
            new_subq, v_qc = self.generate_step(
                current_context, 
                rtc_encoder, 
                tokenizer, 
                device
            )
            
            if not new_subq:
                break
                
            sub_questions.append(new_subq)
            current_context += f" [SEP] {new_subq}" 
            
            with torch.no_grad():
                termination_logit = self.rtc.predictor(v_qc)
                if termination_logit.argmax() == 1:  
                    break
        
        return sub_questions

def train_nsg(model, train_loader, val_loader, rtc_encoder, epochs=3, lr=5e-5):
    device = next(model.parameters()).device
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.bart.config.pad_token_id)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                v_qc, _ = rtc_encoder(input_ids, attention_mask)
            
            prefix_e, prefix_d = model.prefix_injector(v_qc)
            
            model.bart.encoder.prefix = prefix_e.unsqueeze(1)
            model.bart.decoder.prefix = prefix_d.unsqueeze(1)
            
            outputs = model.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict_in_generate=True
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        
        val_loss = evaluate_nsg(model, val_loader, criterion, rtc_encoder, device)
        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    return model

def evaluate_nsg(model, val_loader, criterion, rtc_encoder, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                v_qc, _ = rtc_encoder(input_ids, attention_mask)
            
            prefix_e, prefix_d = model.prefix_injector(v_qc)
            
            model.bart.encoder.prefix = prefix_e.unsqueeze(1)
            model.bart.decoder.prefix = prefix_d.unsqueeze(1)
            
            outputs = model.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict_in_generate=True
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def load_checkpoint(model_class, checkpoint_path, rtc_model_path, device):
    checkpoint = torch.load(checkpoint_path)
    model = model_class.from_pretrained(checkpoint_path)
    
    rtc_checkpoint = torch.load(rtc_model_path)
    model.rtc.load_state_dict(rtc_checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def predict_from_checkpoint(checkpoint_path, question, rtc_model_path, tokenizer, device, max_steps=10):
    rtc = RTC(model_name='bert-base-uncased').to(device)
    rtc.load_state_dict(torch.load(rtc_model_path)['model_state_dict'])
    
    nsg = NSGWithRTC(rtc_model=rtc).to(device)
    
    checkpoint = torch.load(checkpoint_path)
    nsg.load_state_dict(checkpoint['model_state_dict'])
    
    sub_questions = []
    current_context = question
    
    for step in range(max_steps):
        new_subq, v_qc = nsg.generate_step(
            current_context,
            nsg.rtc,
            tokenizer,
            device
        )
        
        if not new_subq:
            break
            
        sub_questions.append(new_subq)
        current_context += f" [SEP] {new_subq}"
        
        with torch.no_grad():
            termination_logit = nsg.rtc.predictor(v_qc)
            if termination_logit.argmax() == 1:  
                break
    
    return sub_questions

if __name__ == "__main__":
    RTC_CHECKPOINT = './rtc_checkpoints/best_model.pth'  
    NSG_CHECKPOINT = './nsg_with_rtc/checkpoint.pth'     
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    results = predict_from_checkpoint(
        checkpoint_path=NSG_CHECKPOINT,
        question="Evaluate the impact of blockchain technology on financial systems",
        rtc_model_path=RTC_CHECKPOINT,
        tokenizer=tokenizer,
        device=device,
        max_steps=10
    )
    
    print("=== Generated Analysis Steps ===")
    for i, subq in enumerate(results, 1):
        print(f"Step {i}: {subq}")
    print("="*30)