import torch
from typing import Any, Dict
from utils import *


class TopKSamplerForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)

    def sample_helper(self, logits, k, temperature):
        # Helper function for the solution.
        values, indices = torch.topk(...) # read the documentation of torch.topk to understand the parameters
        probabilities = torch.softmax(values, dim=-1)
        selected_index = torch.multinomial(probabilities, 1)
        next_token_id = indices[selected_index]
        return next_token_id
    
    @torch.no_grad()
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - It only handles top_k => 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_k (int): the number of highest probability vocabulary tokens 
                         to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################

        # Prepare the encoder and decoder inputs as well as their attention masks
        model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        # score = 0.0

        # Loop over maximum amount of tokens, which can be stopped earlier by 
        # encountering an eos token
        for t in range(...):
            # Sample from the top-k dsitribution for the next token
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            logits = ...
            new_token_id = ... # you can use the sample_helper function to get the new token id

            # Prepare the next inputs to the model with new_token_id
            model_inputs = self.prepare_next_inputs(
                model_inputs = model_inputs,
                new_token_id = new_token_id
            )
            
            # Early stopping: if EOS token, stop decoding
            if new_token_id == self.eos_token_id:
                ...
        
        # Return the sequence generated
        return model_inputs["input_ids"]


class TopPSamplerForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)

    def sample_helper(self, logits, p, temperature):
        # Helper function for the solution.

        # Sort the scaled logits and get cumulative probs
        sorted_logits, sorted_indices = ...
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # print(cumulative_probs)
        torch.set_printoptions(profile="full")

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p

        # Ensure that there is at least one token being sampled!
        # Shift the indices to the right by one, so that the first token
        #   at which cumulative probability exceeds p is not removed
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Mask out the tokens to be removed
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Sample from the remaining distribution
        probabilities = torch.softmax(sorted_logits, dim=-1)
        selected_index = torch.multinomial(probabilities, 1)
        selected_token = sorted_indices[selected_index]
        return selected_token
    
    @torch.no_grad()
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p *or higher*.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) 
        then samples the top scoring token.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################

        # Prepare the encoder and decoder inputs as well as their attention masks
        model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        # score = 0.0

        # Loop over maximum amount of tokens, which can be stopped earlier by 
        # encountering an eos token
        for _ in range(...):
            # Sample from the top-p dsitribution for the next token
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            logits = ...
            new_token_id = ... # you can use the sample_helper function to get the new token id

            # Prepare the next inputs to the model with new_token_id
            model_inputs = self.prepare_next_inputs(
                model_inputs = model_inputs,
                new_token_id = new_token_id
            )

            # Early stopping: if EOS token, stop decoding
            if new_token_id == self.eos_token_id:
                ...
        
        # Return the sequence generated
        return model_inputs["input_ids"]


def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "vicgalle/gpt2-alpaca-gpt4"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()