import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import re
from peft import LoraConfig, get_peft_model

# Load preprocessed financial dataset
news_df = pd.read_csv("yahoo_finance_news.csv")
stocks_df = pd.read_csv("AAPL_stock_data.csv")  # Example stock data file

# Extract date from URL if available
def extract_date_from_url(url):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    return match.group(1) if match else None

if "url" in news_df.columns:
    news_df["date"] = news_df["url"].apply(lambda x: extract_date_from_url(str(x)))
    news_df["date"] = pd.to_datetime(news_df["date"], errors='coerce', utc=True)
else:
    print("❌ Error: 'url' column missing in news_df")
    print("Available columns in news_df:", news_df.columns)
    exit()

stocks_df["Date"] = pd.to_datetime(stocks_df["Date"], errors='coerce', utc=True)

# If no valid dates were extracted, assign the closest stock market date
if news_df["date"].isna().sum() > 0:
    news_df["date"] = stocks_df["Date"].min()

# Print available columns before merging
print("Columns in news_df:", news_df.columns)
print("Columns in stocks_df:", stocks_df.columns)

# Merge stock price data with news headlines
merged_df = pd.merge(news_df, stocks_df, left_on="date", right_on="Date", how="left")
merged_df = merged_df.drop(columns=["date", "Date"], errors='ignore')

# Define sentiment label based on stock % change
if "Close" in merged_df.columns:
    merged_df["pct_change"] = merged_df["Close"].pct_change()
    merged_df["sentiment"] = merged_df["pct_change"].apply(lambda x: "positive" if x > 0.02 else ("negative" if x < -0.02 else "neutral"))
    merged_df.drop(columns=["pct_change"], inplace=True)
else:
    print("❌ Error: 'Close' column missing in merged_df")
    print("Available columns in merged_df:", merged_df.columns)
    exit()

# Remove any rows with missing values
merged_df = merged_df.dropna()

# Convert to Hugging Face dataset format
def format_for_training(example):
    return {"text": f"Financial news: {example['headline']}. Sentiment: {example['sentiment']}"}

dataset = Dataset.from_pandas(merged_df[["headline", "sentiment"]])
dataset = dataset.map(format_for_training)

# Load Falcon model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA configuration
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./falcon-financial-model",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,  # Reduce batch size to prevent memory issues
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True  # Mixed precision training for efficiency
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./falcon-financial-model")
tokenizer.save_pretrained("./falcon-financial-model")

print("Fine-tuning complete. Model saved.")

