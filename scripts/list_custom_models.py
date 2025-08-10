#!/usr/bin/env python3
"""
列出所有可用的custom模型配置
"""
import json
import os
import sys

def list_custom_models():
    config_file = "configs/custom_models.json"
    
    if not os.path.exists(config_file):
        print("❌ JSON配置文件不存在: configs/custom_models.json")
        print("💡 当前使用 .env 配置方式")
        print("")
        print("📋 .env 配置参数:")
        print(f"   custom_code: {os.getenv('custom_code', 'Not set')}")
        print(f"   model_path: {os.getenv('model_path', 'Not set')}")
        print(f"   custom_class: {os.getenv('custom_class', 'Not set')}")
        return
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        custom_models = config.get("custom_models", {})
        
        if not custom_models:
            print("⚠️  JSON配置文件存在但没有配置任何模型")
            return
        
        print("🧠 可用的Custom模型:")
        print("=" * 50)
        
        for model_name, model_config in custom_models.items():
            print(f"📋 模型名: {model_name}")
            if "description" in model_config:
                print(f"   描述: {model_config['description']}")
            print(f"   代码路径: {model_config['custom_code']}")
            print(f"   模型路径: {model_config['model_path']}")
            print(f"   模型类: {model_config['custom_class']}")
            print("")
        
        print("💡 使用方法:")
        print("在脚本中设置:")
        print('model_provider="custom_model"')
        print("models=(")
        for model_name in custom_models.keys():
            print(f' "{model_name}"')
        print(")")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON配置文件格式错误: {e}")
    except Exception as e:
        print(f"❌ 读取配置文件时出错: {e}")

if __name__ == "__main__":
    list_custom_models()