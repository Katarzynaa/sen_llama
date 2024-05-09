import os

from datasets import load_dataset
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

data_files = {"train": "train1.jsonl", "test": "test1.jsonl"}

model_name = "meta-llama/Llama-2-7b-hf"
outdir = "./Model"


def load_model(output_dir: str, model_name: str, device_map=None) -> (AutoPeftModelForCausalLM, AutoTokenizer):
    if device_map is None:
        device_map = {"": 0}
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir + "/final_checkpoint", device_map=device_map,
                                                     torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_sentiment(text: str, model, tokenizer) -> str:
    # model=model.to("cuda")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                             max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    print("OUTPUTS")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate(output_dir: str, base_model_name: str):
    model, tokenizer = load_model(output_dir, base_model_name)
    senentr = load_dataset("json", data_files=data_files, split="test")

    true_senti = []
    pred_senti = []
    for headline in senentr:
        print(headline)
        text = headline["text"].split("### Output:")
        if text[1].replace(" ", "") != "Unknown":
            true_senti.append(text[1].replace(" ", ""))
            # print(text)
            output = get_sentiment(text[0], model, tokenizer)
            print(output)
            sentiment = output.split("###")
            if len(sentiment) >= 4:
                sentiment = sentiment[3].replace(" Output:", "")
            else:
                sentiment = "Neutral"
            pred_senti.append(sentiment.replace(" ", ""
                                                     ""))
            print(sentiment)

    report = classification_report(true_senti, pred_senti)
    print(report)


    with open("./Results/metrics.txt", "w") as outfile:
        outfile.write(report)
    cm = confusion_matrix(true_senti, pred_senti)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("./Results/model_results.png", dpi=120)


def finetune(base_model_name, output_dir):
    senentr = load_dataset("Data", data_files=data_files, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, )
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        report_to="none"
    )

    max_seq_length = 512

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=senentr,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)


finetune(outdir, model_name)

evaluate(outdir, model_name)
