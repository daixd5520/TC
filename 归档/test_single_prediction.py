#!/usr/bin/env python3
"""
测试单个预测的脚本
"""

import json
from cot_t5_classification import CoTT5Classifier, set_seed

def test_single_prediction():
    """测试单个预测"""
    
    # 设置随机种子
    set_seed(42)
    
    # 初始化分类器
    print("正在初始化分类器...")
    classifier = CoTT5Classifier(model_path="./CoT-T5-3B", seed=42)
    
    # 测试数据
    test_text = """Predictive index for optimizing empiric treatment of gram-negative bacteremia.
    In a survey of 296 episodes of gram-negative bacteremia in 286 patients (aged 13-99 years), 
    four clinical variables were found to predict both significantly and independently the subsequent 
    isolation of a multiresistant strain; hospital acquisition of the infection, antibiotic treatment 
    before the bacteremic episode, endotracheal intubation, and thermal trauma as the cause of hospitalization.
    These variables were combined in an index that served to classify the patients into four groups 
    with an increasing prevalence of multiresistant strains, Pseudomonas isolates, and isolates resistant 
    to each of the antibiotic drugs in common use."""
    
    labels = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", 
              "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", 
              "C21", "C22", "C23"]
    
    print(f"测试文本长度: {len(classifier.tokenizer.encode(test_text))} tokens")
    
    # 测试prompt准备
    prompt = classifier.prepare_prompt(test_text, labels)
    prompt_tokens = len(classifier.tokenizer.encode(prompt))
    print(f"Prompt长度: {prompt_tokens} tokens")
    print(f"Prompt内容:\n{prompt}")
    
    # 进行预测
    print("\n进行预测...")
    result = classifier.predict(test_text, labels)
    
    print(f"\n预测结果:")
    print(f"预测标签: {result['label']}")
    print(f"完整输出: {result['full_prediction']}")

if __name__ == "__main__":
    test_single_prediction() 