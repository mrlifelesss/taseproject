from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_id="dicta-il/dictalm2.0"):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tokenizer, model, device

def generate_text(tokenizer, model, device, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model, device = load_model()
    prompt = "שלום, מה אתה יודע לספר לי על מדינת ישראל?"
    generated = generate_text(tokenizer, model, device, prompt)
    print("\nGenerated Text:\n")
    print(generated)
    
    # Optionally save output
    with open("../outputs/generation_output.txt", "w", encoding="utf-8") as f:
        f.write(generated)
