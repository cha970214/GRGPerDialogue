# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import re
from typing import List, Optional, Dict
import fire

from llama import Llama, Dialog

def parse_llama2_persona(generated_text: str) -> Dict[str, List[str]]:
    """
    Llama2가 생성한 결과 문자열(generated_text)에서
    각 Speaker의 persona를 추출해 딕셔너리 형태로 반환합니다.

    예) 입력 (예시):
        Sure! Here are the extracted persona.
        Speaker 1's persona:
        * has a health issue with blood sugar.
        Speaker 2's persona:
        * doesn't have time for sports.

        출력:
        {
          "Speaker 1's persona": ["has a health issue with blood sugar."],
          "Speaker 2's persona": ["doesn't have time for sports."]
        }
    """
    lines = generated_text.splitlines()
    speaker_personas: Dict[str, List[str]] = {}

    current_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match_speaker = re.match(r"^(Speaker\s+\d+'s persona):", line, re.IGNORECASE)
        if match_speaker:
            current_speaker = match_speaker.group(1) 
            if current_speaker not in speaker_personas:
                speaker_personas[current_speaker] = []
            continue

        if line.startswith("*"):
            persona_item = line[1:].strip()
            if current_speaker:
                speaker_personas[current_speaker].append(persona_item)

    return speaker_personas


def main(
    train_txt_path: str,  #
    ckpt_dir: str,       
    tokenizer_path: str,  
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Llama2를 이용해 previous_dialogs에서 각 화자의 페르소나를 추출하되,
    'one-shot example'의 예시 대화를 좀 더 길게 구성한 코드 예시.
    + 후처리(생성 결과 -> JSON 저장) 추가
    """

    one_shot_instruction = (
        "Extract each speaker's persona based on the dialogue.\n"
        "The extracted persona should be sentence(s) like: '<subject> is related with <object>'.\n"
        "You may list multiple persona statements if needed.\n"
    )

    one_shot_dialog_sample = (
        "Example Dialogue:\n"
        "Speaker 1: Hey, do you drink coffee?\n"
        "Speaker 2: I'm a total coffee addict. I have at least 3 cups a day.\n"
        "Speaker 1: Wow, that's a lot. I prefer tea. Coffee makes me too anxious.\n"
        "Speaker 2: Really? Tea just doesn't wake me up as much. But it's healthier!\n"
        "Speaker 1: Yeah, exactly. I also love going to the gym to stay in shape.\n"
        "Speaker 2: That’s cool. I try to jog on weekends, but I usually oversleep.\n"
        "Speaker 1: I never skip my morning routine. It helps me feel energized.\n"
    )

    one_shot_human_annotated_persona = (
        "From this example dialogue, we can infer:\n"
        "Speaker 1's persona:\n"
        "    - dislikes coffee\n"
        "    - prefers tea\n"
        "    - consistently works out in the morning\n\n"
        "Speaker 2's persona:\n"
        "    - loves coffee\n"
        "    - struggles to wake up early\n"
        "    - jogs on weekends but oversleeps\n"
    )

    def build_one_shot_prompt(actual_dialog_str: str) -> str:
        """
        '원-샷 예제' + '실제 대화'를 하나의 User Prompt로 합쳐주는 함수
        """
        return (
            f"{one_shot_instruction}\n\n"
            f"{one_shot_dialog_sample}\n"
            f"{one_shot_human_annotated_persona}\n"
            "---------------------------\n"
            "Now do the same with the following dialogue:\n\n"
            f"{actual_dialog_str}"
        )

    with open(train_txt_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)


    previous_dialogs = train_data.get("previous_dialogs", [])


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    persona_extraction_dialogs: List[Dialog] = []

    for i, pd in enumerate(previous_dialogs):
        dialog_list = pd.get("dialog", [])
        if not dialog_list:
            continue

        # 실제 대화 문자열 구성
        conversation_str = "\n".join(
            [f"{d['id']}: {d['text']}" for d in dialog_list]
        )


        user_prompt = build_one_shot_prompt(conversation_str)

        persona_extraction_dialogs.append(
            [
                {"role": "user", "content": user_prompt}
            ]
        )

    # 추론할 대화가 없으면 종료
    if not persona_extraction_dialogs:
        print("No previous_dialogs found in train.txt.")
        return


    results = generator.chat_completion(
        persona_extraction_dialogs,  # type: ignore
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )

    all_persona_data = {}

    for i, result in enumerate(results):

        generation_text = result["generation"]["content"]

        print(f"=== [Conversation {i+1}] Persona Extraction ===")
        print("-- User Prompt --")
        print(persona_extraction_dialogs[i][0]["content"])
        print("\n-- Model's Persona Extraction --")
        print(generation_text)
        print("====================================\n")


        persona_dict = parse_llama2_persona(generation_text)


        all_persona_data[f"conversation_{i+1}"] = persona_dict


    file_name = "persona_result.json"
    with open(file_name, "w", encoding="utf-8") as fp:
        json.dump(all_persona_data, fp, indent=2, ensure_ascii=False)

    print(f"[Post-Processing] Persona data saved to {file_name}")


if __name__ == "__main__":
    fire.Fire(main)
