from decoding import *
from sampling import *

################################################################################
# NOTE: Caution - do not modify this file!! 
#       If you do, it won't affect your grading, but you might not get 
#       the right results when testing with the notebook.
################################################################################

def hello_W6P2():
    """Helper function to check that functions load properly."""
    print('Hello, this is Part 2 of Week 6 exercise! You successfully linked your directory to Colab.')

class colors:
    """Color class for printing."""
    default='\033[0m'
    red = '\033[31m'
    green = '\033[32m'

def cond_print(condition):
    return "✅" if condition else "❌"

def single_result_seq_print(
        inputs,
        result_ids,
        tokenizer,
        verbose,
        with_color
    ):
    """
    Sequence printer for single outputs.
    """
    if verbose:
        if with_color:
            input_text_ids = inputs["input_ids"][0]
            generated_text_ids = result_ids[0][len(input_text_ids):]
            print(
                "Generated sequence: \n" + \
                colors.red + \
                tokenizer.decode(input_text_ids, skip_special_tokens=False) + \
                colors.green + \
                tokenizer.decode(generated_text_ids, skip_special_tokens=False) + \
                colors.default
            )
        else:
            print("Generated sequence:\n" + tokenizer.batch_decode(result_ids, skip_special_tokens=False)[0])
        print("Output shape: ", result_ids.shape)
        
def summary_print(our_result_ids, hf_result_ids):
    """
    Summary printer for single outputs.
    """
    print("-" * 20)
    print("~ Summary ~")
    id_match = torch.equal(our_result_ids, hf_result_ids)
    length_match = our_result_ids.shape == hf_result_ids.shape
    print("Output IDs matching: ", cond_print(id_match))
    print("Output shape matching: ", cond_print(length_match))
    print("\n")

def multi_result_seq_print(
        inputs,
        result_dict,
        tokenizer,
        verbose,
        with_color
    ):
    """
    Sequence printer for beam outputs.
    """
    seq_scores = enumerate(zip(result_dict["sequences_scores"], result_dict["sequences"]))
    for i, (score, seq) in seq_scores:
        print("{}. score: {}".format(i + 1, score))
        if verbose:
            if with_color:
                input_text_ids = inputs["input_ids"][0]
                generated_text_ids = seq[len(input_text_ids):]
                print(
                    "{}. Generated sequence:\n".format(i + 1) + \
                    colors.red + \
                    tokenizer.decode(input_text_ids, skip_special_tokens=False) + \
                    colors.green + \
                    tokenizer.decode(generated_text_ids, skip_special_tokens=False) + \
                    colors.default
                )
            else:
                print("{}. Generated sequence:\n{}\n".format(i + 1, tokenizer.batch_decode(seq, skip_special_tokens=False)))
    print("Output shape: ", result_dict["sequences"].shape)
    
    
def multi_summary_print(
        num_return_sequences,
        our_result_dict,
        hf_result_dict
    ):
    """
    Summary printer for beam outputs.
    """
    print("-" * 20)
    print("~ Summary ~")
    our_result_dict["sequences"] = our_result_dict["sequences"].to(torch.device("cpu"))
    hf_result_dict["sequences"] = hf_result_dict["sequences"].to(torch.device("cpu"))
    id_match_list = [torch.equal(our_result_dict["sequences"][i], hf_result_dict["sequences"][i]) for i in range(num_return_sequences)]
    length_match = our_result_dict["sequences"].shape == hf_result_dict["sequences"].shape
    score_match_list = [abs(m - h) < 0.1 for m, h in zip(hf_result_dict["sequences_scores"].tolist(), our_result_dict["sequences_scores"])]
    print("Output IDs matching: ", [cond_print(id_match) for id_match in id_match_list])
    print("Output shape matching: ", cond_print(length_match))
    print("Scores approximately close (to 0.1 decimal): ", [cond_print(score_match) for score_match in score_match_list])
    print("\n")


@torch.no_grad()
def greedy_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    verbose: bool = True,
    with_color: bool = False
):
    """
    Greedy tester.
    """
    # 1) Load the decoder
    print("%" * 80)
    print("Greedy Tests")
    print("%" * 80)
    greedy_decoder = GreedySearchDecoderForCausalLM(model=model, tokenizer=tokenizer)

    for title, inputs in all_inputs:
        print("-" * 50)
        print("Input: ", title)
        print("-" * 50)

        print("~ Your Implementation ~")
        torch.manual_seed(42)
        our_result_ids = greedy_decoder.search(
            inputs=inputs,
            max_new_tokens=max_new_tokens
        )
        if our_result_ids is None:
            print("Input constraint encountered. Exiting...")
            exit()
        single_result_seq_print(
            inputs=inputs, 
            result_ids=our_result_ids, 
            tokenizer=tokenizer, 
            verbose=verbose, 
            with_color=with_color
        )

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        torch.manual_seed(42)
        hf_result_ids = greedy_decoder.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            # num_beams=1, # NOTE: these arguments are by default set like this
            # length_penalty=0.0,
            # early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        single_result_seq_print(
            inputs=inputs, 
            result_ids=hf_result_ids, 
            tokenizer=tokenizer, 
            verbose=verbose, 
            with_color=with_color
        )
        
        summary_print(our_result_ids, hf_result_ids)


@torch.no_grad()
def beam_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: int,
    num_return_sequences: int,
    verbose: bool = True,
    with_color: bool = False
):
    """
    Beam tester.
    """
    # 1) Load the decoder
    print("%" * 80)
    print("Beam Tests")
    print("%" * 80)
    beam_decoder = BeamSearchDecoderForCausalLM(model=model, tokenizer=tokenizer)

    # 2) Run it on the 3 examples
    for title, inputs in all_inputs:
        print("-" * 50)
        print("Input: ", title)
        print("-" * 50)

        print("~ Your Implementation ~")
        torch.manual_seed(42)
        our_result_dict = beam_decoder.search(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
        )
        if our_result_dict is None:
            print("Input constraint encountered. Exiting...")
            exit()
        multi_result_seq_print(
            inputs=inputs,
            result_dict=our_result_dict,
            tokenizer=tokenizer,
            verbose=verbose,
            with_color=with_color
        )

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        torch.manual_seed(42)
        hf_result_dict = beam_decoder.model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            num_beams=num_beams,
            early_stopping=True,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            return_dict_in_generate=True,
            num_beam_groups=1,
            constraints=None,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        multi_result_seq_print(
            inputs=inputs,
            result_dict=hf_result_dict,
            tokenizer=tokenizer,
            verbose=verbose,
            with_color=with_color
        )
        
        multi_summary_print(
            num_return_sequences=num_return_sequences, 
            our_result_dict=our_result_dict,
            hf_result_dict=hf_result_dict
        )


@torch.no_grad()
def top_k_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int,
    verbose: bool = True,
    with_color: bool = False
):
    """
    Top-k tester.
    """
    # 1) Load the decoder
    print("%" * 80)
    print("Top-k Tests")
    print("%" * 80)
    top_k_sampler = TopKSamplerForCausalLM(model=model, tokenizer=tokenizer)

    # 2) Run it on the 3 examples
    for title, inputs in all_inputs:
        print("-" * 50)
        print("Input: ", title)
        print("-" * 50)

        print("~ Your Implementation ~")
        torch.manual_seed(seed)
        our_result_ids = top_k_sampler.sample(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature
        )
        if our_result_ids is None:
            print("Input constraint encountered. Exiting...")
            exit()
        single_result_seq_print(
            inputs=inputs, 
            result_ids=our_result_ids, 
            tokenizer=tokenizer, 
            verbose=verbose, 
            with_color=with_color
        )

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        torch.manual_seed(seed)
        hf_result_ids = model.generate(
            **inputs, 
            do_sample=True, 
            max_new_tokens=max_new_tokens, 
            # num_beams=1, # NOTE: these arguments are by default set like this
            # length_penalty=0.0,
            early_stopping=True,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        single_result_seq_print(
            inputs=inputs, 
            result_ids=hf_result_ids, 
            tokenizer=tokenizer, 
            verbose=verbose, 
            with_color=with_color
        )
        
        # NOTE: removing summary print to not create confusion: 
        #       the resulting IDs and shapes can be different due to 
        #       implementation details
        # summary_print(our_result_ids, hf_result_ids)


@torch.no_grad()
def top_p_test(
    model,
    tokenizer,
    all_inputs,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    seed: int,
    verbose: bool = True,
    with_color: bool = False
):
    """
    Top-p tester.
    """
    # 1) Load the decoder
    print("%" * 80)
    print("Top-p Tests")
    print("%" * 80)
    top_p_sampler = TopPSamplerForCausalLM(model=model, tokenizer=tokenizer)

    # 2) Run it on the 3 examples
    for title, inputs in all_inputs:
        print("-" * 50)
        print("Input: ", title)
        print("-" * 50)

        print("~ Your Implementation ~")
        torch.manual_seed(seed)
        our_result_ids = top_p_sampler.sample(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        if our_result_ids is None:
            print("Input constraint encountered. Exiting...")
            exit()
        single_result_seq_print(
            inputs=inputs, 
            result_ids=our_result_ids, 
            tokenizer=tokenizer, 
            verbose=verbose, 
            with_color=with_color
        )

        print("-" * 20)
        print("~ Huggingface Implementation ~")
        torch.manual_seed(seed)
        hf_result_ids = top_p_sampler.model.generate(
            **inputs, 
            do_sample=True, 
            max_new_tokens=max_new_tokens,
            # num_beams=1, # NOTE: these arguments are by default set like this
            # length_penalty=0.0,
            early_stopping=True,
            top_p=top_p,
            top_k=0, # deactivate top_k sampling
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        single_result_seq_print(
            inputs=inputs, 
            result_ids=hf_result_ids, 
            tokenizer=tokenizer, 
            verbose=verbose, 
            with_color=with_color
        )
        
        # NOTE: removing summary print to not create confusion: 
        #       the resulting IDs and shapes can be different due to 
        #       implementation details
        # summary_print(our_result_ids, hf_result_ids)