import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Model and tokenizer
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")

# Apply LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Load and format dataset
data = load_dataset("json", data_files="hr_qa.jsonl")["train"].train_test_split(test_size=0.1, seed=42)

def format(example):
    return {
        "formatted_text": f"<start_of_turn>user\n{example['input']}<end_of_turn>\n"
                          f"<start_of_turn>model\n{example['output']}<end_of_turn>"
    }
train_data = data["train"].map(format)
eval_data = data["test"].map(format)

# Tokenization
def tokenize(example):
    tokens = tokenizer(example["formatted_text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
train_ds = train_data.map(tokenize, remove_columns=train_data.column_names)
eval_ds = eval_data.map(tokenize, remove_columns=eval_data.column_names)

# Training arguments
args = TrainingArguments(
    output_dir="gemma_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=1e-5,
    save_steps=500,
    logging_steps=100,
    no_cuda=True,
    report_to="none",
    save_total_limit=1,
    remove_unused_columns=False
)

# Loss printing callback
class PrintLossCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n🚀 Starting Epoch {int(state.epoch) + 1 if state.epoch is not None else '?'}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ Finished Epoch {int(state.epoch) + 1 if state.epoch is not None else '?'}")
        if self.trainer:
            metrics = self.trainer.evaluate()
            train_loss = next((log["loss"] for log in reversed(state.log_history) if "loss" in log), "N/A")
            print(f"📉 Training Loss: {train_loss}")
            print(f"📊 Validation Loss: {metrics.get('eval_loss', 'N/A')}")

# Collator & trainer
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator
)

# Add callback with trainer reference
trainer.add_callback(PrintLossCallback(trainer))

# Start training
trainer.train()
