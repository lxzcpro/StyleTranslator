#!/usr/bin/env python3
"""
使用 StyleDetector 加载并测试中文 / 英文 BERT 模型
基于 bert_test2.py 的正确加载方式实现
"""

import torch
from transformers import AutoTokenizer
from style_detector.model.model import StyleDetector  # 使用正确的导入方式
import yaml
import os


# ===========================
# 配置文件加载
# ===========================
def load_config(config_path='config.yaml'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None


def load_model(ckpt_path):
    """加载模型（基于bert_test2.py的正确实现）"""
    print(f"正在从checkpoint加载模型: {ckpt_path}")
    
    # 直接使用 StyleDetector 类加载 checkpoint
    model = StyleDetector.load_from_checkpoint(ckpt_path)
    
    # 设置为推理模式
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"✅ 模型加载成功，运行设备: {device}")
    return model


def predict_text(model, tokenizer, text):
    """模型推理函数（基于bert_test2.py的实现）"""
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
    
    return predicted_class, probabilities.cpu().numpy()


# ===========================
# StyleDetector 模型测试逻辑
# ===========================
def test_style_detector(model_path, model_name, style_types):
    """测试StyleDetector模型"""
    print(f"\n{'=' * 60}")
    print(f"测试模型：{model_name}")
    print(f"模型路径: {model_path}")
    print(f"风格类别: {style_types}")
    print(f"{'=' * 60}")

    try:
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"❌ 错误: 模型路径不存在: {model_path}")
            return False

        # 1. 加载模型
        model = load_model(model_path)
        
        # 2. 加载分词器（从模型的超参数中获取模型名称）
        print("\n正在加载分词器...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)
            print(f"✅ 分词器加载成功: {model.hparams.model_name}")
        except Exception as e:
            print(f"❌ 加载分词器失败: {e}")
            return False

        # 3. 测试输入文本
        if "中文" in model_name or "chinese" in model_name.lower():
            test_cases = [
                ("法律风格", "当事人承诺遵守本合同项下的全部义务，任何违约行为均视为实质性违反，并依法承担相应法律责任。"),
                ("新闻风格", "据有关部门透露，为应对客流增长，城市公共交通系统将于下月启动新一轮升级改造。"),
                ("文学风格", "黄昏的光线在街巷间缓缓流淌，仿佛一层轻纱，为这座静谧的城市添上了温柔的色彩。"),
                ("科研论文风格", "实验结果表明，将多模态特征融入模型结构能够在多项评测任务中显著提升系统的稳健性。")
            ]
        else:
            test_cases = [
                ("法律风格", "The party hereby acknowledges that any breach of the obligations stipulated in this Agreement shall constitute a material violation subject to remedies permitted under applicable law."),
                ("新闻风格", "According to officials, the city's public transportation system will undergo a major upgrade next month to address increasing commuter demand."),
                ("文学风格", "The dusk settled like a soft veil over the quiet town, and every fading ray of light seemed to breathe its own wistful farewell."),
                ("科研论文风格", "Our findings demonstrate that integrating multi-modal features significantly improves the robustness of the proposed model across diverse evaluation benchmarks.")
            ]

        # 4. 执行推理
        print("\n测试模型推理...")
        for style_type, text in test_cases:
            predicted_class, probabilities = predict_text(model, tokenizer, text)
            predicted_style = style_types[predicted_class] if predicted_class < len(style_types) else f"未知类别{predicted_class}"
            
            print(f"\n【{style_type}】")
            print(f"文本: '{text}'")
            print(f"预测类别: {predicted_class} ({predicted_style})")
            print("各类别概率分布：")
            for i, (style, prob) in enumerate(zip(style_types, probabilities)):
                print(f"  - {style}: {prob:.8f}")
            print("-" * 60)

        print(f"\n🎉 {model_name} 模型测试完成")
        return True

    except Exception as e:
        print(f"❌ {model_name} 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return False


# ===========================
# 主函数
# ===========================
def main():
    """主函数"""
    print("BERT 风格分类器（StyleDetector）测试开始")
    print("=" * 60)

    # 加载配置文件
    config = load_config()
    if not config:
        return

    # 获取奖励配置
    reward_cfg = config.get("reward", {})
    
    # 获取模型路径和风格类型
    zh_path = reward_cfg.get("chinese_bert_path", "")
    en_path = reward_cfg.get("english_bert_path", "")
    style_types = reward_cfg.get("style_types", [])

    print("读取配置：")
    print(f"- 中文模型路径: {zh_path}")
    print(f"- 英文模型路径: {en_path}")
    print(f"- 风格类别: {style_types}")

    # 测试中文模型
    zh_ok = False
    if zh_path and os.path.exists(zh_path):
        zh_ok = test_style_detector(zh_path, "中文BERT (StyleDetector)", style_types)
    else:
        print(f"❌ 中文模型路径不存在或为空: {zh_path}")

    # 测试英文模型
    en_ok = False
    if en_path and os.path.exists(en_path):
        en_ok = test_style_detector(en_path, "英文BERT (StyleDetector)", style_types)
    else:
        print(f"❌ 英文模型路径不存在或为空: {en_path}")

    # 输出测试结果总结
    print("\n====== 测试结果汇总 ======")
    print(f"中文模型: {'通过 ✔' if zh_ok else '失败 ✘'}")
    print(f"英文模型: {'通过 ✔' if en_ok else '失败 ✘'}")
    
    if zh_ok and en_ok:
        print("🎉 所有模型测试通过！可以正常使用真实BERT模型进行风格奖励计算。")
    else:
        print("⚠️ 部分模型测试失败，请检查模型路径和格式。")


if __name__ == "__main__":
    main()
