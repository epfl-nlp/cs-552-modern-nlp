import torch
from typing import Any, Dict
from utils import *


class GreedySearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the search function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """This function generates sequences of token ids with self.model (which has a language modeling head) using greedy decoding. 
        
        - That is, you always pick the next token with the highest score / probability.
        - Stopping criteria: If the next token is an EOS (end-of-sentence) token, you should stop decoding. Otherwise, stop after generating `max_new_tokens`.
        - This function only handles inputs of batch size = 1.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1, generated_seq_len)
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
        
        # To pass inputs to the model and get the logits, simply use `self.model(**inputs)`. This recomputes the past decoder hidden states at each decoding step.
        # Next week (via solutions), we'll show you how to use the cached decoder hidden states to speed up the process. 
        # It is just for efficiency and not required for this exercise.
        ########################################################################

        # Prepare the decoder inputs as well as their attention masks
        model_inputs = self.prepare_next_inputs(model_inputs=inputs) # take a look at this function in utils.py

        # Loop over maximum amount of tokens, which can be stopped earlier by 
        # encountering an eos token
        for t in range(...):
            # Get logprobs for the next token
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            logits = ...
            log_probs = ...
            
            new_token_id = ...

            # Prepare the next inputs to the model with new_token_id
            model_inputs = self.prepare_next_inputs(
                model_inputs = model_inputs,
                new_token_id = new_token_id,
            )

            # Early stopping: if EOS token, stop decoding
            if new_token_id == self.eos_token_id:
                ...
        
        # Return the sequence generated
        print(model_inputs["input_ids"].device)
        return model_inputs["input_ids"]

class BeamSearchDecoderForCausalLM(GeneratorForCausalLM):
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
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
    ) -> dict: 
        """Generates sequences of token ids with self.model, 
        (which has a language modeling head) using beam search. 
        
        - This means that given a probability distribution over the possible next tokens and 
        a beam width (here `num_beams`), it needs to keep track of the most probable 
        `num_beams` candidates. (Hint: use log probabilities!)
        - Stopping criteria: If the next token is an EOS (end-of-sentence) token, you should stop decoding for that beam. Otherwise, stop after generating `max_new_tokens`.
        - This function only handles inputs of batch size = 1.
        - It only handles beam size > 1.
        - It includes a length_penalty variable that controls the score assigned 
            to a long generation. This is implemented by exponentiating the amount 
            of newly generated tokens to this value. Then, divide the score which 
            can be calculated as the sum of the log probabilities so far.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional): 
                exponential penalty to the length that is used with beam-based generation. 
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. 
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor) 
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence, 
                          you can fill the rest of the tensor row with the pad token ID
                    - "sequences_scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs, 
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
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
        
        # tracks of current beam scores
        beams = [{
            'score': 0.0, 
            'length_penalized_score': 0.0,
            'model_inputs': model_inputs,
            'length': 0.0
        }]

        # Loop over maximum amount of tokens, which can be stopped earlier when 
        # ALL beams have encountered an eos token
        for t in range(...):
            # Keep track of candidates num_beams depth in this round of time step
            candidates = []
            for b in ...:
                # If last token was EOS automatically add to candidates
                # Otherwise carry on with score calculation
                if b["model_inputs"]["input_ids"][0][-1] == self.eos_token_id:
                    ...
                else:
                    # Get logprobs for the next token
                    with torch.no_grad():
                        model_inputs = b["model_inputs"]
                        outputs = self.model(**model_inputs)
                    logits = ...
                    log_probs = ...
                    topk_log_probs, topk_tokens = torch.topk(...)

                    # Loop over top k=num_beams tokens, build their next input and calculate their score
                    for log_prob, new_token_id in zip(topk_log_probs, topk_tokens):
                        new_model_inputs = self.prepare_next_inputs(
                            model_inputs = b["model_inputs"],
                            new_token_id = new_token_id
                        )
                        new_beam = {
                            'score': b['score'] + log_prob.item(),
                            'length': b['length'] + 1.0,
                            'model_inputs': new_model_inputs,
                        }

                        # Keep the penalty score separate from the raw sequence scores
                        tot_length = new_beam['length']
                        new_beam.update({'length_penalized_score': ...}) # simply set this to new_beam['score'] for the without-length-penalty case

                        # Add new candidate beam to candidates list
                        candidates.append(new_beam)

            # Sort candidates by score in descending order
            candidates = sorted(candidates, key=lambda x: x['length_penalized_score'], reverse=True)

            # Select top k=beam_size candidates to become new beam
            beams = candidates[:num_beams]

            # Check if all beams end with end token, stop decoding.
            all_beams_over = True
            for b in beams:
                if b["model_inputs"]["input_ids"][0][-1] != self.eos_token_id:
                    all_beams_over = False
                    break
            if all_beams_over:
                ...
        
        # Fit all the generated sequence into one tensor in the order from best to worst
        # Pad the sentences that are shorter than max_length
        max_beam_len = 0
        # print(len(beams[:num_return_sequences]))
        for input_ids in [b["model_inputs"]["input_ids"] for b in beams[:num_return_sequences]]:
            max_beam_len = max(max_beam_len, input_ids.shape[-1])
        # print(max_beam_len)
        # print(self.pad_token_id)
        return_input_ids = torch.zeros(num_return_sequences, max_beam_len, dtype=torch.long)
        # print(return_input_ids)
        return_input_ids.fill_(value=self.pad_token_id)
        for i in range(num_return_sequences):
            curr_beam = beams[i]["model_inputs"]["input_ids"][0]
            for j in range(len(curr_beam)):
                return_input_ids[i][j] = curr_beam[j]
        
        # Return the sequences with their respective score in the score order.
        return {
            "sequences": return_input_ids,
            "sequences_scores": [b["length_penalized_score"] for b in beams[:num_return_sequences]]
        } 


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