# Modified from https://github.com/kztakemoto/mmllm/blob/main/chatmodel.py
# See Ahmad et al. 2025

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration, BitsAndBytesConfig, AutoConfig

class ChatModel:
    def __init__(self, model, max_new_tokens=None, do_sample=False, load_path=None, dropout=None):
        self.model = model
        print(f'dropout:{dropout}')
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        if "deepseek" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/{}".format(self.model),
            )
            
            config = AutoConfig.from_pretrained(f"deepseek-ai/{self.model}")
            if dropout is not None:
                config.attention_dropout = dropout
            
            self.generator = AutoModelForCausalLM.from_pretrained(
                f"deepseek-ai/{self.model}",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                config=config,
            )
        
        elif "qwen" in self.model.lower() or "qwq" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/{}".format(self.model),
            )
            
            config = AutoConfig.from_pretrained(f"Qwen/{self.model}")
            if dropout is not None:
                config.attention_dropout = dropout
                
            if "72b" in self.model.lower():
                    self.generator = AutoModelForCausalLM.from_pretrained(
                        f"Qwen/{self.model}",
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        device_map="auto",
                        config=config
                    )

            else:
                self.generator = AutoModelForCausalLM.from_pretrained(
                    f"Qwen/{self.model}",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    config=config
                )

        elif "llama" in self.model.lower():
            if "llama-3" in self.model.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"meta-llama/{self.model}",
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            
                config = AutoConfig.from_pretrained(f"meta-llama/{self.model}")
                if dropout is not None:
                    config.attention_dropout = dropout
                    
                if "70b" in self.model.lower():
                    self.generator = AutoModelForCausalLM.from_pretrained(
                        f"meta-llama/{self.model}",
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        device_map="auto",
                        config=config,
                    )

                else:
                    self.generator = AutoModelForCausalLM.from_pretrained(
                        f"meta-llama/{self.model}",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        config=config,
                    )
            else:
                from llama import Llama
                self.generator = Llama.build(
                    ckpt_dir=f"./{self.model}/",
                    tokenizer_path=f"./{self.model}/tokenizer.model",
                    max_seq_len=512,
                    max_batch_size=1,
                )
                
        elif "vicuna" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "lmsys/{}".format(self.model),
                use_fast=False,
            )
            
            config = AutoConfig.from_pretrained("lmsys/{}".format(self.model))
            if dropout is not None:
                config.attention_dropout = dropout
                
            self.generator = AutoModelForCausalLM.from_pretrained(
                "lmsys/{}".format(self.model),
                torch_dtype=torch.float16,
                device_map="auto",
                config=config,
            )
            
        elif "gemma" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/{}".format(self.model),
                use_fast=False,
            )
            
            if "gemma-3" in self.model.lower():
                config = AutoConfig.from_pretrained("google/{}".format(self.model))
                if dropout is not None:
                    config.text_config.attention_dropout = dropout

                self.generator = Gemma3ForConditionalGeneration.from_pretrained(
                    "google/{}".format(self.model),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    config=config,
                )
            else:
                config = AutoConfig.from_pretrained("google/{}".format(self.model))
                if dropout is not None:
                    config.attention_dropout = dropout
                self.generator = AutoModelForCausalLM.from_pretrained(
                    "google/{}".format(self.model),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    config=config,
                )
                
        elif "mistral" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/{}".format(self.model),
                use_fast=False,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                        
            config = AutoConfig.from_pretrained("mistralai/{}".format(self.model))
            if dropout is not None:
                config.attention_dropout = dropout

            self.generator = AutoModelForCausalLM.from_pretrained(
                "mistralai/{}".format(self.model),
                torch_dtype=torch.float16,
                device_map="auto",
                config=config,
            )
            
        elif "command" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "CohereForAI/{}".format(self.model),
            )
                        
            config = AutoConfig.from_pretrained("CohereForAI/{}".format(self.model))
            if dropout is not None:
                config.attention_dropout = dropout
                
            self.generator = AutoModelForCausalLM.from_pretrained(
                "CohereForAI/{}".format(self.model),
                device_map="auto",
                config=config,
            )
            
        elif "phi-" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/{}".format(self.model),
            )
                        
            config = AutoConfig.from_pretrained("microsoft/{}".format(self.model))
            if dropout is not None:
                config.attention_dropout = dropout
                
            if "moe" in self.model.lower():
                self.generator = AutoModelForCausalLM.from_pretrained(
                    "microsoft/{}".format(self.model),
                    device_map="auto", 
                    torch_dtype="auto", 
                    trust_remote_code=False,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    config=config,
                )
            else:
                self.generator = AutoModelForCausalLM.from_pretrained(
                    "microsoft/{}".format(self.model),
                    device_map="auto", 
                    torch_dtype="auto", 
                    trust_remote_code=False,
                    config=config,
                )

        else:
            raise ValueError("unsupprted model")
            
        if load_path is not None:
            self.generator.load_state_dict(torch.load(load_path))

    def chat(self, system_prompt, user_prompt, assistant_prompt='', max_new_tokens=None):
        if max_new_tokens is not None:
            self.max_new_tokens=max_new_tokens
        
        if "deepseek" in self.model.lower():
            return self.chat_deepseek(system_prompt, user_prompt, assistant_prompt)
        elif "qwen" in self.model.lower() or "qwq" in self.model.lower():
            return self.chat_qwen(system_prompt, user_prompt, assistant_prompt)
        elif "llama" in self.model.lower():
            if "llama-3" in self.model.lower():
                return self.chat_llama3_hf(system_prompt, user_prompt, assistant_prompt)
            elif "llama-2" in self.model.lower():
                return self.chat_llama2_hf(system_prompt, user_prompt, assistant_prompt)
            # else:
            #     return self.chat_llama(system_prompt, user_prompt, assistant_prompt)
        elif "vicuna" in self.model.lower():
            return self.chat_vicuna(system_prompt, user_prompt, assistant_prompt)
        elif "gemma" in self.model.lower():
            return self.chat_gemma(system_prompt, user_prompt, assistant_prompt)
        elif "mistral" in self.model.lower():
            return self.chat_mistral(system_prompt, user_prompt, assistant_prompt)
        elif "command" in self.model.lower():
            return self.chat_command(system_prompt, user_prompt, assistant_prompt)
        elif "phi" in self.model.lower():
            return self.chat_phi(system_prompt, user_prompt, assistant_prompt)

#     def chat_llama(self, system_prompt, user_prompt, assistant_prompt):
#         dialogs = [
#             [
#                 {"role": "system", "content": f"Please respond to binary questions.\n\n{system_prompt}"},
#                 {"role": "user", "content": user_prompt},
#                 {"role": "assistant", "content": assistant_prompt},
#             ],
#         ]
#         response = self.generator.chat_completion(
#             dialogs,  # type: ignore
#             max_gen_len=128,
#             # temperature=0.6,
#             # top_p=0.9,
#             output_scores=True,        # Enable logits output
#             return_dict_in_generate=True,  # Return a dict with sequences and scores
#         )

#         return response[0]['generation']['content']
    
    def chat_llama3_hf(self, system_prompt, user_prompt, assistant_prompt):
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nPlease respond to binary questions.\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>{assistant_prompt}"
        
        
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=256 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample, # To get greedy deterministic response
                # temperature=0,
                # top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response
    
    def chat_llama2_hf(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]{assistant_prompt}"
        
        
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=256 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample, # To get greedy deterministic response
                # temperature=0.6,
                # top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response
    
    def chat_deepseek(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"<｜begin▁of▁sentence｜>Please respond to binary questions. {system_prompt}<｜User｜>{user_prompt}<｜Assistant｜>{assistant_prompt}"
        
        
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=2048 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample, # To get greedy deterministic response
                # temperature=0.6,
                # top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>"),
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response

    def chat_qwen(self, system_prompt, user_prompt, assistant_prompt):
        if "qwen3" in self.model.lower():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
                # enable_thinking=True # [TODO] Thinking parsing not applied yet
            )

            prompt = text + f'<|assistant|>{assistant_prompt}'
            
            
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,             
                    max_new_tokens=32768 if not self.max_new_tokens else self.max_new_tokens,
                    do_sample=self.do_sample,  # To get greedy deterministic response
                    output_scores=True,        # Enable logits output
                    return_dict_in_generate=True,  # Return a dict with sequences and scores
                )

            response = dict(inputs=inputs, outputs=outputs)
            return response

        else:
            prompt = f"<|im_start|>system\nPlease respond to binary questions. {system_prompt}<|im_end|>\n\n<|im_start|>user\n{user_prompt}<|im_end|>\n\n<|im_start|>assistant{assistant_prompt}"
            
            
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=2048 if not self.max_new_tokens else self.max_new_tokens,
                    do_sample=self.do_sample,  # To get greedy deterministic response
                    # temperature=0.6,
                    # top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,        # Enable logits output
                    return_dict_in_generate=True,  # Return a dict with sequences and scores
                )
                
            response = dict(inputs=inputs, outputs=outputs)
            return response

    def chat_vicuna(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"USER: Please respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}\n\nASSISTANT:{assistant_prompt}"
        
        
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=512 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample,  # To get greedy deterministic response
                # temperature=0.7,
                # top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response

    def chat_gemma(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"<bos><start_of_turn>user\nPlease respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{assistant_prompt}"

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=512 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample,  # To get greedy deterministic response
                # temperature=0.7,
                # top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response

    def chat_mistral(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"<s>[INST] Please respond to binary questions.\n\n{system_prompt}\n\n{user_prompt} [/INST]{assistant_prompt}"
        
        
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=512 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample,  # To get greedy deterministic response
                # temperature=0.7,
                # top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response

    def chat_command(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Please respond to binary questions.\n\n{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{assistant_prompt}"
        
        
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=100 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample,  # To get greedy deterministic response
                # temperature=0.3,
                # top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )

        response = dict(inputs=inputs, outputs=outputs)
        return response
        
    def chat_phi(self, system_prompt, user_prompt, assistant_prompt):
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>{assistant_prompt}"
        

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=100 if not self.max_new_tokens else self.max_new_tokens,
                do_sample=self.do_sample,  # To get greedy deterministic response
                # temperature=0.7,
                # top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"),
                output_scores=True,        # Enable logits output
                return_dict_in_generate=True,  # Return a dict with sequences and scores
            )


        response = dict(inputs=inputs, outputs=outputs)
        return response

    
    def get_text(self, response, output_only=True):
        inputs = response['inputs']
        outputs = response['outputs']
        if output_only:
            decoded = self.tokenizer.decode(outputs.sequences.tolist()[0][inputs.input_ids.size(1):])
        else:
            decoded = self.tokenizer.decode(outputs.sequences.tolist()[0])
        return str(decoded)
    
    def get_probs(self, response, token_position=0, topk=2):
        inputs = response['inputs']
        outputs = response['outputs']
        
        assert token_position<len(outputs.scores), 'idx should be less then the size of outputs.scores' 
        token_logits = outputs.scores[token_position]  # Logits for the first (and only) new token
        token_probs = torch.softmax(token_logits, dim=-1) # Convert to probabilities (softmax)
        
        top_probs, top_indices = torch.topk(token_probs, k=topk, dim=-1)
        top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices[token_position]]
        return {token:prob.item() for token, prob in zip(top_tokens, top_probs[token_position])}
      
    def get_probs_with_tokens(self, response, token1='1', token2='2'):
        token_id_1 = self.tokenizer.convert_tokens_to_ids(token1)
        token_id_2 = self.tokenizer.convert_tokens_to_ids(token2)

        inputs = response['inputs']
        outputs = response['outputs']
        
        token_logits = outputs.scores[0]  # Logits for the first (and only) new token
        token_probs = torch.softmax(token_logits, dim=-1) # Convert to probabilities (softmax)
        binary_probs = token_probs[:, [token_id_1, token_id_2]]
        p1, p2 = binary_probs[0]
        return p1, p2