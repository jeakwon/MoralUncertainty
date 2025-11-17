import pandas as pd
import random
import os
from tqdm import tqdm

from generate_moral_machine_scenarios import generate_moral_machine_scenarios
from chatapi import ChatBotManager
from chatmodel import ChatModel

import argparse

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Qwen3-0.6B', type=str)
parser.add_argument('--nb_scenarios', default=1000, type=int)
parser.add_argument('--random_seed', default=123, type=int)
parser.add_argument('--save_dir', default='../results', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--dropout', default=None, type=float)
args = parser.parse_args()

# Ensure save_dir exists
os.makedirs(args.save_dir, exist_ok=True)

# load LLM model (API)
if any(s.lower() in args.model.lower() for s in ["gpt", "o1", "o3", "o4", "gemini", "claude", "palm", "deepseek-chat", "deepseek-reasoner", "grok"]):
  chat_model = ChatBotManager(model=args.model)
elif any(s.lower() in args.model.lower() for s in ["llama", "vicuna", "gemma", "mistral", "command", "phi", "qwen", "deepseek"]):
  chat_model = ChatModel(model=args.model, load_path=args.load_path, dropout=args.dropout)
  if args.dropout is not None:
    chat_model.generator.train() # to make dropout alive. gradient is not generated during chat
    print(f'Applied dropout on model. dropout={args.dropout}')
else:
  raise ValueError("Unsupported model")

# obtain LLM responses
file_name = os.path.join(args.save_dir, str(args.nb_scenarios), args.model, f"results_{args.nb_scenarios}_scenarios_seed{args.random_seed}_{args.model}.pickle")
random.seed(args.random_seed)
scenario_info_list = []
for i in tqdm(range(args.nb_scenarios)):
  # scenario dimension
  dimension = random.choice(["species", "social_value", "gender", "age", "fitness", "utilitarianism"])
  #dimension = "random"
  # Interventionism #########
  is_interventionism = random.choice([True, False])
  # Relationship to vehicle #########
  is_in_car = random.choice([True, False])
  # Concern for law #########
  is_law = random.choice([True, False])
  
  # generate a scenario
  system_content, user_content, scenario_info = generate_moral_machine_scenarios(dimension, is_in_car, is_interventionism, is_law)

  # obtain chatgpt response
  response = chat_model.chat(system_content, user_content)
  scenario_info['chat_response'] = response
  #print(scenario_info)

  scenario_info_list.append(scenario_info)

  if (i+1) % 100 == 0:
    df = pd.DataFrame(scenario_info_list)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Ensure model dir exists
    df.to_pickle(file_name)

df = pd.DataFrame(scenario_info_list)
os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Ensure model dir exists
df.to_pickle(file_name)