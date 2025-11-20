import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

def convert_lightning_ckpt_to_hf(
    ckpt_path,
    pretrained_name,
    output_dir,
    num_labels=4
):
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise ValueError("❌ The checkpoint does not contain 'state_dict' (not a Lightning ckpt?)")

    state_dict = ckpt["state_dict"]
    print("Checkpoint keys loaded:", len(state_dict))

    # Load pretrained BERT config
    print(f"Loading HuggingFace BERT base: {pretrained_name}")
    config = BertConfig.from_pretrained(pretrained_name)
    config.num_labels = num_labels

    # Load pretrained BERT model
    model = BertForSequenceClassification.from_pretrained(pretrained_name, config=config)

    # Remove prefix "model." "encoder." etc.
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ["model.", "encoder.", "bert."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned_state_dict[new_k] = v

    print("Loading cleaned state_dict into model...")
    model.load_state_dict(cleaned_state_dict, strict=False)

    print(f"Saving HF-style model to {output_dir}")
    model.save_pretrained(output_dir)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_name)
    tokenizer.save_pretrained(output_dir)

    print("✅ Conversion complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    # ---- MODIFY THESE ----
    ckpt_path = r"G:\ECE\ECE684_NLP\final\models\berts\english_style_detector_final.ckpt"               # 修改 ✔
    pretrained_name = "bert-base-cased"        # 或 "bert-base-cased"
    output_dir = r"G:\ECE\ECE684_NLP\final\models\berts\english"            # 输出目录 ✔
    num_labels = 4                               # 根据你的任务修改（风格分类 4 类）

    convert_lightning_ckpt_to_hf(ckpt_path, pretrained_name, output_dir, num_labels)
