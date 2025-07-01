#!/usr/bin/env python3
"""
测试token长度设置的脚本
"""

import json
from cot_t5_classification import CoTT5Classifier, set_seed

def test_token_length():
    """测试token长度设置"""
    
    # 设置随机种子
    set_seed(42)
    
    # 初始化分类器
    print("正在初始化分类器...")
    classifier = CoTT5Classifier(model_path="./CoT-T5-3B", seed=42)
    
    # 测试一个较长的文本
    test_text = """Predictive index for optimizing empiric treatment of gram-negative bacteremia.
    In a survey of 296 episodes of gram-negative bacteremia in 286 patients (aged 13-99 years), 
    four clinical variables were found to predict both significantly and independently the subsequent 
    isolation of a multiresistant strain; hospital acquisition of the infection, antibiotic treatment 
    before the bacteremic episode, endotracheal intubation, and thermal trauma as the cause of hospitalization.
    These variables were combined in an index that served to classify the patients into four groups 
    with an increasing prevalence of multiresistant strains, Pseudomonas isolates, and isolates resistant 
    to each of the antibiotic drugs in common use. For example, the percentage of isolates susceptible 
    to cefuroxime in the four groups were 79%, 56%, 34% and 25%, and to gentamicin, 89%, 79%, 46%, 
    and 33% (P less than .001 for both comparisons). The performance of the index was validated 
    in a second group of 144 episodes of gram-negative bacteremia. The index kept its discriminative power.
    Compared with the prescriptions of the attending physicians, the index could probably have improved 
    empiric antibiotic treatment in 24% of patients."""
    
    labels = ["C01", "C02", "C03", "C04", "C05"]
    
    print(f"\n测试文本长度: {len(classifier.tokenizer.encode(test_text))} tokens")
    print(f"最大文本长度: {classifier.max_text_tokens} tokens")
    print(f"最大输入长度: {classifier.max_input_tokens} tokens")
    
    # 测试prompt准备
    prompt = classifier.prepare_prompt(test_text, labels)
    prompt_tokens = len(classifier.tokenizer.encode(prompt))
    print(f"完整prompt长度: {prompt_tokens} tokens")
    
    # 测试一个真实的数据样本
    print("\n测试真实数据样本...")
    with open("./ohsumed_converted/ohsumed_test_1000.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample = data[0]
    sample_text = sample["input"]
    sample_labels = [sample["category"]]
    
    print(f"样本文本长度: {len(classifier.tokenizer.encode(sample_text))} tokens")
    sample_prompt = classifier.prepare_prompt(sample_text, sample_labels)
    sample_prompt_tokens = len(classifier.tokenizer.encode(sample_prompt))
    print(f"样本prompt长度: {sample_prompt_tokens} tokens")
    
    # 统计数据集中的文本长度分布
    print("\n分析数据集文本长度分布...")
    text_lengths = []
    for i, item in enumerate(data[:100]):  # 只分析前100个样本
        text_length = len(classifier.tokenizer.encode(item["input"]))
        text_lengths.append(text_length)
    
    text_lengths.sort()
    print(f"前100个样本的文本长度统计:")
    print(f"  最短: {text_lengths[0]} tokens")
    print(f"  最长: {text_lengths[-1]} tokens")
    print(f"  中位数: {text_lengths[len(text_lengths)//2]} tokens")
    print(f"  平均: {sum(text_lengths)/len(text_lengths):.1f} tokens")
    
    # 统计需要截断的样本数量
    truncated_count = sum(1 for length in text_lengths if length > classifier.max_text_tokens)
    print(f"  需要截断的样本数: {truncated_count}/{len(text_lengths)} ({truncated_count/len(text_lengths)*100:.1f}%)")

if __name__ == "__main__":
    test_token_length() 