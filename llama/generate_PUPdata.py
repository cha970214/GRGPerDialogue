import json
import re
import random
import fire
from typing import List, Dict

try:
    from llama import Llama
except ImportError:
    print("Please install or import your Llama2 library correctly.")

def generate_persona_with_llama2(
    utterance: str,
    generator: "Llama",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 64
) -> str:


    prompt = (
        "Below is a single utterance from a speaker.\n"
        "Please extract a short persona or set of persona statements that describe the speaker's\n"
        "preferences, hobbies, background, or traits reflected in the utterance.\n\n"
        f"Utterance: \"{utterance}\"\n\n"
        "Persona:"
    )

    results = generator.text_completion(
        prompts=[prompt],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    persona_text = results[0]["generation"].strip()

    return persona_text


def remove_irrelevant_words(text: str) -> str:

    text = re.sub(r"(?i)speaker\s*1", "", text)
    text = re.sub(r"(?i)speaker\s*2", "", text)
    return text.strip()


def build_pup_from_previous(
    previous_utterances: List[str],
    generator: "Llama"
) -> List[Dict[str, str]]:

    pup_data = []
    for i in range(len(previous_utterances) - 1):
        c_i = previous_utterances[i]
        c_next = previous_utterances[i + 1]

        raw_persona = generate_persona_with_llama2(
            utterance=c_next,
            generator=generator,
            temperature=0.6,
            top_p=0.9,
            max_gen_len=64
        )

        cleaned_persona = remove_irrelevant_words(raw_persona)

        pup_data.append({
            "utterance": c_i,
            "persona": cleaned_persona
        })

    return pup_data


def main(
    train_txt_path: str = "train.txt",
    ckpt_dir: str = "<PATH_TO_LLAMA_WEIGHTS>",
    tokenizer_path: str = "<PATH_TO_TOKENIZER>",
    output_prefix: str = "pup_previous",
    train_ratio: float = 0.8,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: int = 64,
):
    from llama import Llama
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    with open(train_txt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    previous_utterances = data.get("previous", [])
    if not previous_utterances:
        print("No 'previous' found in train.txt. Check file structure.")
        return


    pup_data = []
    pup_data = build_pup_from_previous(previous_utterances, generator)
    random.shuffle(pup_data)
    split_idx = int(len(pup_data) * train_ratio)
    train_data = pup_data[:split_idx]
    val_data = pup_data[split_idx:]

    train_file = f"{output_prefix}_train.json"
    val_file = f"{output_prefix}_val.json"

    with open(train_file, "w", encoding="utf-8") as f_tr:
        json.dump(train_data, f_tr, ensure_ascii=False, indent=2)
    with open(val_file, "w", encoding="utf-8") as f_val:
        json.dump(val_data, f_val, ensure_ascii=False, indent=2)

    print(f"[INFO] Created PUP dataset from 'previous' in {train_txt_path}")
    print(f" - Total pairs: {len(pup_data)}")
    print(f" - Train: {len(train_data)}, Val: {len(val_data)}")
    print(f" - Saved to: {train_file}, {val_file}")


if __name__ == "__main__":
    fire.Fire(main)
