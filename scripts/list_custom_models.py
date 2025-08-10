#!/usr/bin/env python3
"""
List all available custom model configurations
"""
import json
import os
import sys

def list_custom_models():
    config_file = "configs/custom_models.json"
    
    if not os.path.exists(config_file):
        print("❌ JSON config file does not exist: configs/custom_models.json")
        print("💡 Currently using .env configuration method")
        print("")
        print("📋 .env configuration parameters:")
        print(f"   custom_code: {os.getenv('custom_code', 'Not set')}")
        print(f"   model_path: {os.getenv('model_path', 'Not set')}")
        print(f"   custom_class: {os.getenv('custom_class', 'Not set')}")
        return
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        custom_models = config.get("custom_models", {})
        
        if not custom_models:
            print("⚠️  JSON config file exists but no models are configured")
            return
        
        print("🧠 Available Custom models:")
        print("=" * 50)
        
        for model_name, model_config in custom_models.items():
            print(f"📋 Model name: {model_name}")
            if "description" in model_config:
                print(f"   Description: {model_config['description']}")
            print(f"   Code path: {model_config['custom_code']}")
            print(f"   Model path: {model_config['model_path']}")
            print(f"   Model class: {model_config['custom_class']}")
            print("")
        
        print("💡 Usage:")
        print("Set in script:")
        print('model_provider="custom_model"')
        print("models=(")
        for model_name in custom_models.keys():
            print(f' "{model_name}"')
        print(")")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON config file format error: {e}")
    except Exception as e:
        print(f"❌ Error reading config file: {e}")

if __name__ == "__main__":
    list_custom_models()