from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen3:
    def __init__(self, model_path: str):
        """
        Initialize your custom model.
        
        Args:
            model_path (str): Path to your model files/weights
        """
        self.model_path = model_path
        
        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def invoke(self, messages: list[dict]) -> str:
        """
        Perform inference on the given messages.
        
        Args:
            messages (list[dict]): List of message dictionaries
                Each message should have the format:
                {"role": "system|user|assistant", "content": "message content"}
        
        Returns:
            str: JSON-formatted string containing the model's response
        """
        # print("================================================")
        # print("Messages:", messages)
        # print("================================================")

        # Prepare the model input        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        # Parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        # print("Qwen3 Response:", response)
        
        return response

# from peft import PeftModel

# class Llama3:
#     def __init__(self, model_path: str):
#         """
#         Initialize Llama3.2 model.
        
#         Args:
#             model_path (str): Path to your model files/weights
#         """
#         self.model_path = model_path
        
#         # Load the tokenizer and the model
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True
#         )
        
#         # Ensure padding token is set
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
  
#     def invoke(self, messages: list[dict]) -> str:
#         """
#         Perform inference on the given messages.
        
#         Args:
#             messages (list[dict]): List of message dictionaries
#                 Each message should have the format:
#                 {"role": "system|user|assistant", "content": "message content"}
        
#         Returns:
#             str: Generated response from the model
#         """
#         print("================================================")
#         print("Messages:", messages)
#         print("================================================")

#         # Apply chat template to convert messages to text format
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         # Tokenize the input text
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

#         # Generate response
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens=256,
#                 temperature=0.7,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
        
#         # Extract only the newly generated tokens
#         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
#         # Decode the response
#         response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
#         print("Llama3.2 Response:", response)
        
#         return response
   
# class Llama3_lora_3B(Llama3):
#     def __init__(self, model_path):
#         base_name = "meta-llama/Llama-3.2-3B-Instruct"
#         self.tokenizer = AutoTokenizer.from_pretrained(base_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             base_name,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True
#         )
#         self.model = PeftModel.from_pretrained(self.model, model_path)
       
# class Llama3_lora_1B(Llama3):
#     def __init__(self, model_path):
#         base_name = "meta-llama/Llama-3.2-1B-Instruct"
#         self.tokenizer = AutoTokenizer.from_pretrained(base_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             base_name,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True
#         )
#         self.model = PeftModel.from_pretrained(self.model, model_path)
