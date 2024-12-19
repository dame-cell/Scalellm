import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self, model_name, **model_kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(self, question, num_return_sequences=1, max_new_tokens=1024, temperature=0.8, system_prompt:str=None, **generate_kwargs):
        if system_prompt is None:
            # PROMPT USE BY THE HUGGINFACE TEAM HERE https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute  
            # TO DO : try to experiment with different prompts for qwen models
            system_prompt = """
            ## Instructions:
            Solve the following math problem efficiently and clearly:

            - **For simple problems (2 steps or fewer):**
              Provide a concise solution with minimal explanation.

            - **For complex problems (3 steps or more):**
              Use this step-by-step format:

              ### Step-by-step format:
              - **Step 1:** [Concise description]
                [Brief explanation and calculations]

              - **Step 2:** [Concise description]
                [Brief explanation and calculations]

            - Regardless of the approach, always conclude with:
              **Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.**
              (Where [answer] is the final number or expression that solves the problem.)
            """
        else:
            system_prompt = system_prompt

        prompt = f"{question}\n{system_prompt}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated_outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            **generate_kwargs
        )

        return generated_outputs