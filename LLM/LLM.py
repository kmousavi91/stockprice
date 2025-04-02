from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

from datasets import load_dataset


# Step 1: Load Pretrained Open-Access Model (Falcon-7B)
model_name = "tiiuae/falcon-7b"

# Ensure memory-efficient loading
torch.cuda.empty_cache()  # Clears GPU memory

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with optimized device management
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Move model properly
model = model.to_empty(device="cuda" if torch.cuda.is_available() else "cpu")

print("Model loaded successfully!")




# Step 2: Apply LoRA Adapter for Efficient Fine-Tuning
lora_config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05, bias="none")  # Smaller LoRA adapters to save memory
model = get_peft_model(model, lora_config)

# Step 3: Load and Preprocess Dataset with Streaming
dataset = load_dataset("ccdv/patent-classification", split="train[:1000]")
train_dataset = dataset.shuffle(seed=42)


# Step 4: Define Training Arguments with Lower Memory Usage
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="no",  # Disable evaluation to avoid errors
    save_strategy="steps",
    save_steps=500,
    learning_rate=3e-5,
    num_train_epochs=2,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    logging_dir="./logs"
)

# Step 5: Train the Model with Optimized Memory Usage
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Use streamed dataset
    tokenizer=tokenizer
)
trainer.train()

# Step 6: Save the Fine-Tuned Model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 7: Load and Use the Fine-Tuned Model for Inference
def preprocess_function(examples):
    if "text" in examples:
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    else:
        raise ValueError("Dataset does not contain a 'text' column. Please check the dataset format.")

dataset = dataset.map(preprocess_function, batched=True)

# Example Usage
response = generate_response("Find patents related to graphene-based transistors.")
print("Generated Response:", response)

