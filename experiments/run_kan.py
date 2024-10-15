import logging

import torch
from datasets import load_dataset
from kan_gpt.model import GPT as KAN_GPT
from transformers import AutoTokenizer, TrainingArguments

# === Define loss logging ===
loss_logger = logging.getLogger("loss_logger")
loss_logger.setLevel(logging.INFO)
fh = logging.FileHandler("training_loss.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)
loss_logger.addHandler(fh)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PARAMS = "31"
MAX_LENGTH = 1024
BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained("Hack90/virus_pythia_31_1024")
model_config = KAN_GPT.get_default_config()
model_config.model_type = "gpt-mini"
model_config.vocab_size = 8
model_config.block_size = MAX_LENGTH
model = KAN_GPT(model_config)

ds_train = load_dataset("Hack90/experiment_one_viral_genomes_train_set")
ds_valid = load_dataset("Hack90/experiment_one_viral_genomes_val_set")
ds_test = load_dataset("Hack90/experiment_one_viral_genomes_test_set")

for ds in [ds_train, ds_valid, ds_test]:
    ds.set_format(type="torch")

tokenizer.pad_token = tokenizer.eos_token

logger.info("Training with loss type: CE")

training_args = TrainingArguments(
    output_dir=f"./virus_KAN_21m_{MAX_LENGTH}_CE",
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=10,
    logging_steps=100,
    logging_dir="./logs",
    learning_rate=0.00005,
    dataloader_num_workers=1,
    dataloader_prefetch_factor=1,
    report_to=[],
)

# === Custom Training Loop ===
step = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()

for _epoch in range(training_args.num_train_epochs):
    for i in range(len(ds_train["train"])):
        optimizer.zero_grad()

        input_id = ds_train["train"]["input_ids"][i]
        input_id_truncated = input_id[:-1].unsqueeze(0)
        shifted = input_id[1:].unsqueeze(0)

        loss = model(input_id_truncated, shifted)
        loss_value = loss[1]
        loss[1].backward()

        # Log the gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_mean = param.grad.mean().item()
        #         loss_logger.info(f"Step: {step}, Gradient mean for {name}: {grad_mean}")

        loss_logger.info(f"Step: {step}, Loss: {loss_value}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        if step % 1000 == 0:
            # Evaluate the model
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for j in range(len(ds_valid["train"])):
                    input_id = ds_valid["train"]["input_ids"][j]
                    input_id_truncated = input_id[:-1].unsqueeze(0)
                    shifted = input_id[1:].unsqueeze(0)

                    loss = model(input_id_truncated, shifted)
                    eval_loss += loss[1]

            eval_loss /= len(ds_valid["train"])
            logger.info(f"Step: {step}, Eval Loss: {eval_loss}")
