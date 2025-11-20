import torch
from transformers import AutoTokenizer
from style_detector.model.model import StyleDetector  # ← 这里保证和你的模型文件名一致

CKPT_PATH = r"G:\ECE\ECE684_NLP\final\models\berts\chinese_style_detector_final.ckpt"

def load_model(ckpt_path):
    print(f"Loading model from checkpoint: {ckpt_path}")

    # 直接使用 LightningModule 类加载 checkpoint
    model = StyleDetector.load_from_checkpoint(ckpt_path)

    # 设置为推理模式
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Model loaded successfully.")
    return model


def predict_text(model, tokenizer, text):
    device = model.device

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, logits.cpu().numpy()


if __name__ == "__main__":
    # -------------------------
    # 1. Load checkpoint model
    # -------------------------
    model = load_model(CKPT_PATH)

    # -------------------------
    # 2. Load pretrained tokenizer
    #    从超参数中自动读取 model_name
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)

    # -------------------------
    # 3. Input text to test
    # -------------------------
    while True:
        text = input("\n请输入文本（或输入 q 退出）：\n")
        if text.strip().lower() == "q":
            break

        pred_class, probs = predict_text(model, tokenizer, text)

        print("\n=== 预测结果 ===")
        print(f"预测类别: {pred_class}")
        print(f"各类别概率: {probs}")
