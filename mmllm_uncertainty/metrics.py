import pandas as pd
from tqdm import tqdm

def get_death_probs(model, moral_machine_scenarios, assistant_prompt='Case ', max_new_tokens=1, topk=None, verbose=False):
    death_probs = []
    for moral_machine_scenario in tqdm(moral_machine_scenarios):
        system_prompt = moral_machine_scenario['system_prompt']
        user_prompt = moral_machine_scenario['user_prompt']
        labels = moral_machine_scenario['labels']
        
        response = model.chat(system_prompt, user_prompt, assistant_prompt=assistant_prompt, max_new_tokens=max_new_tokens)
        text = model.get_text(response)
        p1, p2 = model.get_probs_with_tokens(response, '1', '2')        
        l1, l2 = labels

        death_probs.append( {l1:p1.item(), l2:p2.item()} )
        
        if verbose:
            print(f'{model_name}: Case {text} (Death Probabilty: p1={p1:.1%} [{l1}], p2={p2:.1%} [{l2}])')
    return pd.DataFrame(death_probs)