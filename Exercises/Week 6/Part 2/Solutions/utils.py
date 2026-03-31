import os
import torch
import random
import numpy as np
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

class GeneratorForCausalLM():
    ###########################################################################
    # NOTE: Caution - do not modify the inputs to the helper class, however, feel free
    # to add as many helper functions in this class as you want or to modify the 
    # prepare_decoder_input_ids function :)
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """A helper generator class for decoding and sampling algorithms.
        You are not required to use its fields or the self.prepare_next_inputs function.

        Args:
            model (GPT2LMHeadModel): model to generate with
            tokenizer (AutoTokenizer): corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_max_length = tokenizer.model_max_length
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.model.eval()
        
    ############################################################################
    # NOTE: Don't change the input_constraints function, it's to help you debug!
    ############################################################################
    def input_constraints(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams : int = None,
        num_return_sequences: int = None,
        top_k: int = None
    ):
        """A helper function to let you know that you don't need to handle the 
        certain edge cases.

        Args:
            inputs (dict)
            max_new_tokens (int)
            num_beams (int, optional). Defaults to None.
            num_return_sequences (int, optional). Defaults to None.

        Returns:
            Any: either max_new_tokens or None if not within constraints
        """
        if max_new_tokens < 1:
            print("Generation should be at least 1 token. Returning None.")
            return None
        batch_size = inputs["input_ids"].shape[0]
        if batch_size != 1:
            print(f"Your batch_size={batch_size} but this function only handles batch_size=1. Returning None.")
            return None
        if self.model_max_length < max_new_tokens:
            print("Truncating max_new_tokens = {} to the model's maximum capacity = {}.".format(
                max_new_tokens,
                self.model_max_length))
            max_new_tokens = self.model_max_length
            return max_new_tokens
        # Only concerns beam search
        if num_return_sequences is not None:
            if num_return_sequences > num_beams or num_return_sequences < 2:
                print("num_return_sequences should be more than 1 and less than num_beams.")
                return None
        # Only concerns top-k
        if top_k is not None:
            if top_k < 1:
                print("top_k should be more than 0.")
                return None
        
        # Otherwise return original max_new_tokens
        return max_new_tokens

    @torch.no_grad()
    def prepare_next_inputs(
            self,
            model_inputs: dict,
            new_token_id: torch.Tensor = None,
            returned_past_key_values: torch.Tensor = None,
            use_cache: bool = False,
            use_cuda: bool = False,
        ) -> dict:
        """"A helper function to prepare input ids and their attention mask 
        to be passed during the forward pass of the model. 

        You do not need to use this function but we created this space in case 
        you feel like you have repeated code and would like to make your code 
        more readable!
        
        Feel free to change its arguments as you wish.

        Args:
            model_inputs (dict): the last inputs to the model
            new_token_id (torch.Tensor, optional): the token ID to be added for next inputs. Defaults to None.
            returned_past_key_values (torch.Tensor, optional): cached past_key_values. Defaults to None.
            use_cuda (bool): Whether to move tensors to cuda or not. Defaults to False.

        Returns:
            dict: the next model input dictionary
        """
                
        new_model_inputs = {}
        if new_token_id is None:
            # First step of decoding: may need to move to CUDA
            new_model_inputs.update(model_inputs)
            if use_cuda:
                new_model_inputs["input_ids"] = new_model_inputs["input_ids"].cuda()
                new_model_inputs["attention_mask"] = new_model_inputs["attention_mask"].cuda()
            if use_cache:
                new_model_inputs["past_key_values"] = None
        else:
            # Next steps of decoding:
            if use_cache:
                # If using cached KV states, only input_ids that do not have 
                # their past calculated should be passed as input_ids (i.e. the new_token_id)
                #
                # Input IDs:
                new_model_inputs["input_ids"] = model_inputs["input_ids"].new_ones((1, 1)) 
                new_model_inputs["input_ids"][0][-1] = new_token_id
                #
                #
                # Attention: If cache is used, attention_mask needs to 
                # contain the masking strategy that was used for past_key_values. 
                # In other words, the attention_mask always has to have the length: 
                # len(past_key_values) + len(input_ids)
                new_model_inputs["attention_mask"] = torch.cat([
                    model_inputs["attention_mask"], 
                    model_inputs["attention_mask"].new_ones((1, 1))
                ],dim=-1)
                #
                # Cache: Need to pass as input the cached KVs
                new_model_inputs["past_key_values"] = returned_past_key_values
            else:
                # If not using cached KV states, we need to concatenate 
                # the new token ID and its attention mask to the old ones
                #
                # Input IDs:
                new_model_inputs["input_ids"] = torch.cat([
                    model_inputs["input_ids"],
                    model_inputs["input_ids"].new_ones((1, 1))],
                    dim=-1,
                )
                new_model_inputs["input_ids"][0][-1] = new_token_id
                #
                # Attention:
                new_model_inputs["attention_mask"] = torch.cat([
                    model_inputs["attention_mask"], 
                    model_inputs["attention_mask"].new_ones((1, 1))
                ],dim=-1)
                #
                # Cache:
                # new_model_inputs["past_key_values"] = None
        return new_model_inputs


def load_seed(seed : int):
    """Sets the seed for several packages you may use, in case it's not torch.
    You can also just do torch.manual_seed(seed).

    Args:
        seed (int): the seed number
    """
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: if distributed training/inference
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)