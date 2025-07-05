#!/usr/bin/env python3
"""
CoT生成功能测试脚本
用于验证API返回的CoT结果是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ds_api_v2 import DeepSeekCoTGenerator, test_single_api, test_cot_generation, validate_cot_response

def main():
    """主测试函数"""
    print("=" * 60)
    print("CoT生成功能测试")
    print("=" * 60)
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 单个API测试 (快速)")
    print("2. 完整功能测试 (3个案例)")
    print("3. 自定义测试")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        print("\n运行单个API测试...")
        test_single_api()
        
    elif choice == "2":
        print("\n运行完整功能测试...")
        results = test_cot_generation()
        
        # 显示详细结果
        if results:
            print("\n" + "=" * 60)
            print("详细测试结果:")
            print("=" * 60)
            
            for result in results:
                print(f"\n测试案例 {result['test_id']}: {result['description']}")
                print(f"成功: {'✅' if result.get('success') else '❌'}")
                
                if result.get('success'):
                    validation = result['validation']
                    print(f"  - 响应长度: {validation['response_length']}")
                    print(f"  - 包含正确标签: {'✅' if validation['contains_correct_label'] else '❌'}")
                    print(f"  - 包含步骤分析: {'✅' if validation['has_step_analysis'] else '❌'}")
                    print(f"  - 最终分类正确: {'✅' if validation['final_classification_correct'] else '❌'}")
                    print(f"  - 包含医学分析: {'✅' if validation['has_medical_analysis'] else '❌'}")
                    
                    if validation['issues']:
                        print(f"  - 问题: {', '.join(validation['issues'])}")
                else:
                    print(f"  - 错误: {result.get('error', '未知错误')}")
        
    elif choice == "3":
        print("\n自定义测试模式")
        print("请输入测试文本 (输入 'quit' 退出):")
        
        while True:
            text = input("\n测试文本: ").strip()
            if text.lower() == 'quit':
                break
                
            label = input("正确标签 (如 C14): ").strip()
            if not label:
                print("标签不能为空，跳过")
                continue
            
            print(f"\n测试: {text[:50]}... -> {label}")
            
            try:
                generator = DeepSeekCoTGenerator()
                response = generator.generate_cot_response(text, label, "custom_test")
                
                if response:
                    print("✅ 生成成功")
                    print("响应内容:")
                    print("-" * 50)
                    print(response)
                    print("-" * 50)
                    
                    # 验证
                    validation = validate_cot_response(response, label)
                    print(f"\n验证结果:")
                    print(f"  - 总体有效: {'✅' if validation['is_valid'] else '❌'}")
                    print(f"  - 包含正确标签: {'✅' if validation['contains_correct_label'] else '❌'}")
                    print(f"  - 包含步骤分析: {'✅' if validation['has_step_analysis'] else '❌'}")
                    print(f"  - 最终分类正确: {'✅' if validation['final_classification_correct'] else '❌'}")
                    
                    if validation['issues']:
                        print(f"  - 问题: {', '.join(validation['issues'])}")
                else:
                    print("❌ 生成失败")
                    
            except Exception as e:
                print(f"❌ 测试异常: {e}")
    
    else:
        print("无效选择，退出")

if __name__ == "__main__":
    main() 