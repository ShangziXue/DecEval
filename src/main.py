import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rtc import RTC, load_checkpoint as load_rtc_checkpoint
from subgen import NSGWithRTC, load_checkpoint as load_nsg_checkpoint
from rationale_eval import RSP, load_model as load_rsp_model

def load_llama2_model(model_name='meta-llama/Llama-2-7b-hf'):
    return AutoModelForCausalLM.from_pretrained(model_name)

def format_reasoning_chain(context_chain):
    formatted = []
    for i, (q, a) in enumerate(context_chain, 1):
        formatted.append(f"Step {i}:Q: {q}A: {a}")
    return "".join(formatted)

def main():
    RTC_CHECKPOINT = './nsg_with_rtc/checkpoint.pth'
    NSG_CHECKPOINT = './nsg_with_rtc/checkpoint.pth'
    RSP_CHECKPOINT = './rsp_checkpoints/best_model.pth'
    TEST_QUESTIONS = [
        "Evaluate the impact of blockchain technology on financial systems",
        "Analyze the relationship between climate change and agricultural production"
    ]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_STEPS = 5
    RELEVANCE_THRESHOLD = 0.7

    rtc = load_rtc_checkpoint(RTC_CHECKPOINT, device)
    nsg = load_nsg_checkpoint(NSG_CHECKPOINT, rtc, device)
    rsp_model = load_rsp_model(RSP_CHECKPOINT, device)
    llama_model = load_llama2_model().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

    for question in TEST_QUESTIONS:
        print(f"Processing question: {question}")
        print("="*50)
        
        print("Generating sub-questions...")
        sub_questions = nsg.generate_step(
            current_context=question,
            rtc_encoder=rtc,
            tokenizer=tokenizer,
            device=DEVICE
        )
        print(f"Generated {len(sub_questions)} sub-questions")

        context_chain = []
        full_reasoning = []

        for i, subq in enumerate(sub_questions):
            print(f"{'-'*50}")
            print(f"Sub-question {i+1}/{len(sub_questions)}: {subq}")
            
            input_text = question
            if context_chain:
                input_text += " [SEP] " + " [SEP] ".join([f"Q: {qc}A: {ac}" for qc, ac in context_chain])
            
            inputs = tokenizer(
                input_text,
                return_tensors='pt',
                max_length=1024,
                truncation=True
            ).to(DEVICE)
            
            outputs = llama_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answers.append(answer)
            
            print(f"Generated answer preview: {answer[:100]}...")
            relevance_score = rsp_model.predict_relevance(
                question=question,
                reasoning=answer,
                tokenizer=tokenizer,
                device=DEVICE,
                threshold=RELEVANCE_THRESHOLD
            )
            
            print(f"Relevance Score: {relevance_score:.4f} | Accepted: {relevance_score >= RELEVANCE_THRESHOLD}")
            
            if relevance_score >= RELEVANCE_THRESHOLD:
                formatted_step = f"Step {i+1}:Q: {subq}A: {answer}"
                context_chain.append((subq, answer))
                full_reasoning.append(formatted_step)
            else:
                print("Answer rejected due to low relevance")

        print("" + "="*50)
        print("Full Reasoning Chain:")
        print(format_reasoning_chain(context_chain))
        
        with open(f"reasoning_chain_{question[:20]}.txt", "w") as f:
            f.write(format_reasoning_chain(context_chain))

if __name__ == "__main__":
    main()