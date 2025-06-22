#!/usr/bin/env python3
"""
测试结果复现性的脚本
"""

import json
from cot_t5_classification import CoTT5Classifier, set_seed

def test_reproducibility():
    """测试结果复现性"""
    
    # 测试数据
    test_text = """Predictive index for optimizing empiric treatment of gram-negative bacteremia.
    In a survey of 296 episodes of gram-negative bacteremia in 286 patients (aged 13-99 years), 
    four clinical variables were found to predict both significantly and independently the subsequent 
    isolation of a multiresistant strain; hospital acquisition of the infection, antibiotic treatment 
    before the bacteremic episode, endotracheal intubation, and thermal trauma as the cause of hospitalization."""
    
    labels = ["C01", "C02", "C03", "C04", "C05"]
    
    print("测试结果复现性...")
    print("="*50)
    
    # 运行两次相同的预测
    results = []
    for run in range(2):
        print(f"\n第 {run + 1} 次运行:")
        
        # 设置相同的随机种子
        set_seed(42)
        
        # 初始化分类器
        classifier = CoTT5Classifier(model_path="./CoT-T5-3B", seed=42)
        
        # 进行预测
        result = classifier.predict(test_text, labels)
        
        print(f"  预测标签: {result['label']}")
        print(f"  完整输出: {result['full_prediction'][:100]}...")
        
        results.append(result)
    
    # 检查结果是否一致
    if results[0]['label'] == results[1]['label']:
        print(f"\n✅ 结果一致！两次预测的标签都是: {results[0]['label']}")
    else:
        print(f"\n❌ 结果不一致！")
        print(f"  第一次预测: {results[0]['label']}")
        print(f"  第二次预测: {results[1]['label']}")
    
    # 检查完整输出是否一致
    if results[0]['full_prediction'] == results[1]['full_prediction']:
        print("✅ 完整输出也一致！")
    else:
        print("❌ 完整输出不一致！")
        print(f"  第一次输出: {results[0]['full_prediction'][:100]}...")
        print(f"  第二次输出: {results[1]['full_prediction'][:100]}...")

if __name__ == "__main__":
    test_reproducibility() 