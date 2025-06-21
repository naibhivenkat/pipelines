from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from languages import LANGUAGE_MODELS
import os

def train_model(lang: str, train_path: str, test_path: str):
    model_checkpoint = LANGUAGE_MODELS[lang]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

    dataset = load_dataset('csv', data_files={'train': train_path, 'test': test_path})
    dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir=f'models/{lang}_model',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()
    model.save_pretrained(f'models/{lang}_model')
    tokenizer.save_pretrained(f'models/{lang}_model')