import torch
import os
import pandas as pd
import torch.optim as optim
import numpy as np
import json

from mmllm.chatmodel import ChatModel
from torch.utils.data import Dataset, DataLoader

class ScenarioDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # List of (text, entropy_target) tuples

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]  # Return (text, entropy_target)

def collate_fn(batch, tokenizer):
    texts, targets = zip(*batch)
    inputs = tokenizer(list(texts), add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, padding_side='left')
    targets = torch.tensor(targets)  # Convert to tensor
    return inputs, targets

def load_data(root, H_targets, model_name):
    dfs = {}
    for fname in os.listdir(root):
        if not fname.endswith('.csv'):
            continue
        scenario = fname.replace('.csv', '')
        fpath = os.path.join(root, fname)
        df = pd.read_csv(fpath)
        dfs[scenario] = df
    return dfs

def get_prompt(model_name, system_prompt, user_prompt, assistant_prompt=''):
    if "deepseek" in model_name.lower():
        return f"<｜begin▁of▁sentence｜>Please respond to binary questions. {system_prompt}<｜User｜>{user_prompt}<｜Assistant｜>{assistant_prompt}"
    
    elif "qwen" in model_name.lower() or "qwq" in model_name.lower():
        if "qwen3" in model_name.lower():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            # For Qwen3, we need to apply chat template, but since we're not generating, simulate the prompt
            # Note: For forward pass, we can build the string similarly, but actual template might need tokenizer.apply_chat_template
            # To handle properly, we'll assume string format here; adjust if needed
            text = f"<|im_start|>system\nPlease respond to binary questions. {system_prompt}<|im_end|>\n\n<|im_start|>user\n{user_prompt}<|im_end|>\n\n<|im_start|>assistant{assistant_prompt}"
            return text
        else:
            return f"<|im_start|>system\nPlease respond to binary questions. {system_prompt}<|im_end|>\n\n<|im_start|>user\n{user_prompt}<|im_end|>\n\n<|im_start|>assistant{assistant_prompt}"
    
    elif "llama" in model_name.lower():
        if "llama-3" in model_name.lower():
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nPlease respond to binary questions.\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>{assistant_prompt}"
        elif "llama-2" in model_name.lower():
            return f"<s>[INST] <<SYS>>\nPlease respond to binary questions.\n\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]{assistant_prompt}"
    
    elif "vicuna" in model_name.lower():
        return f"USER: Please respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}\n\nASSISTANT:{assistant_prompt}"
    
    elif "gemma" in model_name.lower():
        return f"<bos><start_of_turn>user\nPlease respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{assistant_prompt}"
    
    elif "mistral" in model_name.lower():
        return f"<s>[INST] Please respond to binary questions.\n\n{system_prompt}\n\n{user_prompt} [/INST]{assistant_prompt}"
    
    elif "command" in model_name.lower():
        return f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Please respond to binary questions.\n\n{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{assistant_prompt}"
    
    elif "phi" in model_name.lower():
        return f"<|system|>\nPlease respond to binary questions.\n\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>{assistant_prompt}"
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def prepare_train_data(train_dfs, H_targets, model_name):
    train_data_list = []
    for scenario, df in train_dfs.items():
        entropy_target = H_targets[model_name][scenario]
        for idx, row in df.iterrows():
            system_prompt = row['system_prompt']
            user_prompt = row['user_prompt']
            assistant_prompt = 'Case '  # Assuming this is fixed as per original
            text = get_prompt(model_name, system_prompt, user_prompt, assistant_prompt)
            train_data_list.append((text, entropy_target))
    return train_data_list

def prepare_test_data(test_dfs, H_targets, model_name):
    test_scenario_datasets = {}
    for scenario, df in test_dfs.items():
        texts = []
        for idx, row in df.iterrows():
            system_prompt = row['system_prompt']
            user_prompt = row['user_prompt']
            assistant_prompt = 'Case '  # Assuming this is fixed
            text = get_prompt(model_name, system_prompt, user_prompt, assistant_prompt)
            texts.append(text)
        
        entropy_target = H_targets[model_name][scenario]
        test_scenario_datasets[scenario] = (texts, entropy_target)
    return test_scenario_datasets

def compute_avg_entropy(model, tokenizer, scenario, texts, entropy_target, batch_size, num_workers, eval_mode=True):
    if eval_mode:
        model.eval()
    else:
        model.train()
    
    dataset = ScenarioDataset([(text, entropy_target) for text in texts])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    all_binary_entropies = []
    
    with torch.no_grad() if eval_mode else torch.enable_grad():
        for batch_inputs, _ in dataloader:
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            outputs = model(**batch_inputs)
            logits = outputs.logits[:, -1, :]

            token_id_1 = tokenizer.convert_tokens_to_ids('1')
            token_id_2 = tokenizer.convert_tokens_to_ids('2')

            selected_logits = logits[:, [token_id_1, token_id_2]]
            binary_probs = torch.softmax(selected_logits, dim=-1)
            p, q = binary_probs.T
            binary_entropy = -p*torch.log2(p) - q*torch.log2(q)
            
            all_binary_entropies.append(binary_entropy)
    
    all_entropies = torch.cat(all_binary_entropies)
    entropy_mean = all_entropies.mean()
    
    if eval_mode:
        model.train()
    
    return entropy_mean.item()

def fine_tune(
        model_name, 
        root_train='mm_dataset/1000', 
        root_test='mm_dataset/100', 
        epochs=3, 
        lr=1e-4, 
        batch_size=5, 
        num_workers=4, 
        save_dir='finetune/',
        h_target_path='H_targets.csv'
    ):
    H_targets = pd.read_csv(h_target_path, index_col=0)
    
    chatmodel = ChatModel(model_name)
    model = chatmodel.generator
    tokenizer = chatmodel.tokenizer
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    train_dfs = load_data(root_train, H_targets, model_name)
    test_dfs = load_data(root_test, H_targets, model_name)
    
    train_data_list = prepare_train_data(train_dfs, H_targets, model_name)
    test_scenario_datasets = prepare_test_data(test_dfs, H_targets, model_name)
    
    # Create model-specific directory
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, 'results.json')
    
    # Results dictionary for tracking
    results = {
        'initial': {
            'correlation': None,
            'scenarios': {}
        },
        'epochs': [],
        'final': {
            'correlation': None,
            'scenarios': {}
        }
    }
    
    # Initial evaluation
    initial_test_entropies = {}
    print("Initial Test Entropies:")
    tgt_list = []
    avg_list = []
    for scenario, (texts, entropy_target) in test_scenario_datasets.items():
        initial_entropy_mean = compute_avg_entropy(model, tokenizer, scenario, texts, entropy_target, batch_size, num_workers)
        initial_test_entropies[scenario] = initial_entropy_mean
        print(f'[Test Scenario:{scenario:>20}] Initial H_avg: {initial_entropy_mean:.4f}, H_tgt: {entropy_target:.4f}, Diff: {abs(initial_entropy_mean - entropy_target):.4f}')
        tgt_list.append(entropy_target)
        avg_list.append(initial_entropy_mean)
        results['initial']['scenarios'][scenario] = {
            'H_avg': initial_entropy_mean,
            'H_tgt': entropy_target,
            'diff': abs(initial_entropy_mean - entropy_target)
        }
    initial_corr = np.corrcoef(tgt_list, avg_list)[0, 1]
    results['initial']['correlation'] = initial_corr
    print(f'Initial Correlation Coefficient (tgt vs avg): {initial_corr:.4f}')
    
    # Training
    train_dataset = ScenarioDataset(train_data_list)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for step, (batch_inputs, batch_targets) in enumerate(train_dataloader):
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            batch_targets = batch_targets.to(model.device)
            outputs = model(**batch_inputs)
            logits = outputs.logits[:, -1, :]

            token_id_1 = tokenizer.convert_tokens_to_ids('1')
            token_id_2 = tokenizer.convert_tokens_to_ids('2')

            selected_logits = logits[:, [token_id_1, token_id_2]]
            binary_probs = torch.softmax(selected_logits, dim=-1)
            p, q = binary_probs.T
            binary_entropy = -p*torch.log2(p) - q*torch.log2(q)
            
            unique_targets = torch.unique(batch_targets)
            mse_losses = []
            for tgt in unique_targets:
                mask = (batch_targets == tgt)
                group_entropy_mean = binary_entropy[mask].mean()
                group_mse = (group_entropy_mean - tgt) ** 2
                mse_losses.append(group_mse)
            
            if mse_losses:
                mse_loss = torch.mean(torch.stack(mse_losses))
            else:
                mse_loss = torch.tensor(0.0, device=model.device)
            
            epoch_losses.append(mse_loss.item())

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
        
        avg_epoch_loss = np.mean(epoch_losses)
        print(f'[Epoch:{epoch+1:>3}] Average Loss: {avg_epoch_loss:.4f}')

        # Epoch evaluation
        print(f"Epoch {epoch+1} Test Entropies:")
        tgt_list = []
        avg_list = []
        epoch_scenarios = {}
        for scenario, (texts, entropy_target) in test_scenario_datasets.items():
            epoch_entropy_mean = compute_avg_entropy(model, tokenizer, scenario, texts, entropy_target, batch_size, num_workers)
            print(f'[Test Scenario:{scenario:>20}, Epoch:{epoch+1:>3}] H_avg: {epoch_entropy_mean:.4f}, H_tgt: {entropy_target:.4f}, Diff: {abs(epoch_entropy_mean - entropy_target):.4f}')
            tgt_list.append(entropy_target)
            avg_list.append(epoch_entropy_mean)
            epoch_scenarios[scenario] = {
                'H_avg': epoch_entropy_mean,
                'H_tgt': entropy_target,
                'diff': abs(epoch_entropy_mean - entropy_target)
            }
        epoch_corr = np.corrcoef(tgt_list, avg_list)[0, 1]
        print(f'Epoch {epoch+1} Correlation Coefficient (tgt vs avg): {epoch_corr:.4f}')
        
        results['epochs'].append({
            'epoch': epoch + 1,
            'avg_loss': avg_epoch_loss,
            'correlation': epoch_corr,
            'scenarios': epoch_scenarios
        })
        
        # Save checkpoint for this epoch
        checkpoint_path = os.path.join(model_dir, f'checkpoint-{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch+1} to {checkpoint_path}")
    
    # Final evaluation
    print("Final Test Entropies and Comparisons:")
    tgt_list = []
    avg_list = []
    for scenario, (texts, entropy_target) in test_scenario_datasets.items():
        final_entropy_mean = compute_avg_entropy(model, tokenizer, scenario, texts, entropy_target, batch_size, num_workers)
        initial_entropy_mean = initial_test_entropies[scenario]
        print(f'[Test Scenario:{scenario:>20}] H_tgt: {entropy_target:.4f}, Final H_avg: {final_entropy_mean:.4f}, Initial H_avg: {initial_entropy_mean:.4f}, Change: {final_entropy_mean - initial_entropy_mean:.4f}, Final Diff: {abs(final_entropy_mean - entropy_target):.4f}')
        tgt_list.append(entropy_target)
        avg_list.append(final_entropy_mean)
        results['final']['scenarios'][scenario] = {
            'H_avg': final_entropy_mean,
            'H_tgt': entropy_target,
            'initial_H_avg': initial_entropy_mean,
            'change': final_entropy_mean - initial_entropy_mean,
            'final_diff': abs(final_entropy_mean - entropy_target)
        }
    final_corr = np.corrcoef(tgt_list, avg_list)[0, 1]
    results['final']['correlation'] = final_corr
    print(f'Final Correlation Coefficient (tgt vs avg): {final_corr:.4f}')
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    return model, tokenizer