from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
EOS = tok.token_to_id("<eos>")
IGNORE = -100


def build_example(prompt, response, max_len=1024):
    # Encode the two halves separately, exactly as the model will see them
    # at inference: the prompt ends at "Assistant:" and the model writes on.
    prompt_ids = [EOS] + tok.encode(f"User: {prompt}\nAssistant:").ids
    resp_ids = tok.encode(" " + response).ids + [EOS]

    ids = prompt_ids + resp_ids
    labels = [IGNORE] * len(prompt_ids) + resp_ids
    return ids[:max_len], labels[:max_len]


if __name__ == "__main__":
    ids, labels = build_example("find a red ball", "Moving toward the red ball.")

    # Check 1: the prompt tokens match what eval.py feeds the model
    check = [EOS] + tok.encode("User: find a red ball\nAssistant:").ids
    assert ids[:len(check)] == check, "prompt tokens differ from inference!"

    # Check 2: show every token and whether the loss is computed on it
    for t, l in zip(ids, labels):
        status = "mask " if l == IGNORE else "TRAIN"
        print(f"{status} {t:6d} {tok.id_to_token(t)!r}")