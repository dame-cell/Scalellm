from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# COPIED FROM THE HUGGINGFACE SEARCH AND LEARN GITHUB REPO
CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902

class RLHFModel:
  def __init__(self,model_name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",**model_kwargs):
    self.model = AutoModelForCausalLM.from_pretrained(model_name,**model_kwargs)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.tokenizer.padding_side = "right"
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model.config.pad_token_id = self.model.config.eos_token_id

    plus_tag_id = self.tokenizer.encode("+")[-1]
    minus_tag_id = self.tokenizer.encode("-")[-1]
    self.candidate_tokens = [plus_tag_id, minus_tag_id]

    
  def score(self,question,outputs,batched:bool,batch_size:int=2):
    if batched:
      return self.score_batched(question,outputs,batch_size)
    else:
      return self.score_single(question,outputs)
    
  # COPIED FROM https://github.com/huggingface/search-and-learn/blob/main/src/sal/models/reward_models.py
  # TO DO: TOO SLOW CAN WE SPEED IT UP ? 
  def _score_single(self, questions: list[str], outputs: list[list[str]]):
      # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
      all_scores = []
      for question, answers in zip(questions, outputs, strict=True):
          all_step_scores = []
          for ans in answers:
              single_step_score = []
              conversation = []
              ans_list = ans.split("\n\n")
              for k in range(len(ans_list)):
                  if k == 0:
                      # TODO: add the system prompt like we did for math shepard?
                      text = question + " " + ans_list[0]
                  else:
                      text = ans_list[k]
                  conversation.append({"content": text, "role": "user"})
                  conversation.append({"content": "+", "role": "assistant"})
                  input_ids = self.tokenizer.apply_chat_template(
                      conversation, return_tensors="pt"
                  ).to(self.model.device)
                  with torch.no_grad():
                      logits = self.model(input_ids).logits[
                          :, -3, self.candidate_tokens
                      ]  # simple version, the +/- is predicted by the '-3' position
                      step_scores = logits.softmax(dim=-1)[
                          :, 0
                      ]  # 0 means the prob of + (1 mean -)
                      # print(scores)
                      single_step_score.append(
                          step_scores[0]
                          .detach()
                          .to("cpu", dtype=torch.float32)
                          .item()
                      )

              all_step_scores.append(single_step_score)
          all_scores.append(all_step_scores)
      return all_scores

  def _score_batched(
      self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
  ):
      # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
      # we need to introduce a dummy special token here for masking.

      special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
      # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
      conversations = []
      conversations2 = []
      for question, answers in zip(questions, outputs, strict=True):
          for ans in answers:
              conversation = []
              conversation2 = []
              ans_list = ans.split("\n\n")
              for k in range(len(ans_list)):
                  if k == 0:
                      text = question + " " + ans_list[0]
                  else:
                      text = ans_list[k]
                  conversation.append({"content": text, "role": "user"})
                  conversation.append({"content": "+", "role": "assistant"})

                  # we track to location of the special token with ки in order to extract the scores
                  conversation2.append({"content": text, "role": "user"})
                  conversation2.append({"content": "ки", "role": "assistant"})

              conversations.append(conversation)
              conversations2.append(conversation2)

      output_scores = []
      for i in range(0, len(conversations), batch_size):
          convs_batch = conversations[i : i + batch_size]
          convs2_batch = conversations2[i : i + batch_size]
          inputs_batch = self.tokenizer.apply_chat_template(
              convs_batch, padding=True, return_tensors="pt"
          ).to(self.model.device)
          inputs2_batch = self.tokenizer.apply_chat_template(
              convs2_batch, padding=True, return_tensors="pt"
          ).to(self.model.device)
          assert inputs_batch.shape == inputs2_batch.shape
          with torch.no_grad():
              logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
              scores = logits.softmax(dim=-1)[
                  :, :, 0
              ]  # 0 means the prob of + (1 mean -)

              for i in range(len(convs_batch)):
                  # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                  step_scores_flat = scores[i, :-1][
                      inputs2_batch[i, 1:] == special_tok_id
                  ].tolist()
                  output_scores.append(step_scores_flat)

      # reshape the output scores to match the input
      reshaped_output_scores = []
      counter = 0
      for question, answers in zip(questions, outputs):
          scores = []
          for answer in answers:
              scores.append(output_scores[counter])
              counter += 1
          reshaped_output_scores.append(scores)

      return reshaped_output_scores