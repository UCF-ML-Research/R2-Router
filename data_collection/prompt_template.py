"""
Prompt template functions for dataset processing and judging.
"""


def build_prompt(golden_answer, answer):
    """
    Build a prompt for judging the correctness of an answer against a golden answer.
    
    Args:
        golden_answer (str): The correct/golden answer
        answer (str): The candidate answer to be judged
        
    Returns:
        str: Formatted prompt for the judge
    """
    return f"""
You are a strict judge.  
Compare the candidate answer with the golden answer(s).  

Give a correctness score from {{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}}, and a brief justification.  

Output strictly in JSON:
{{
    "correctness_score": <your_score>,
    "justification": "<your_reason>"
}}

### Golden Answer(s)
{golden_answer}

Steps you MUST follow:
1. Compare the Candidate Answer with the Golden Answer(s).  
2. If the Candidate Answer is too short, vague, or does not cover key points from the Golden Answer(s),  
   assign a low score.  
3. Only assign 1.0 if the Candidate Answer clearly matches one of the Golden Answer(s).
4. Do not confuse the Golden Answer(s) with the Candidate Answer.

### Candidate Answer
{answer}

Now give your JSON response:
"""


def generate_prompts_with_token_limits(original_prompts, max_tokens):
    """
    Generate prompts with token limits by adding a constraint to the original prompts.
    
    Args:
        original_prompts (list): List of original prompt strings
        max_tokens (int): Maximum number of words allowed in the response. If -1, no limit is applied.
        
    Returns:
        list: List of modified prompts with token limits
    """
    if max_tokens == 0:
        return original_prompts
    
    template = "{original_prompt}\n\nYou have a strict budget of {max_tokens} words. You must answer at most {max_tokens} words!\n Answer:"
    return [
        template.format(max_tokens=max_tokens, original_prompt=p)
        for p in original_prompts
    ]
