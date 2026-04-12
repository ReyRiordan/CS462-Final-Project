import json
import os
import random
import time
import requests
from dotenv import load_dotenv
import re
import numpy as np


load_dotenv()
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
API_KEY = os.getenv("FIREWORKS_API_KEY")


def build_system_prompt(base: str, rubric: dict) -> str:
    rubric_block = {
        "title": rubric["title"],
        "features": rubric["features"]
    }
    return base.replace("{RUBRIC_BLOCK}", json.dumps(rubric_block, ensure_ascii=False))


def generate(model_id, system: str, user: str, logprobs, temperature: float = 0.7) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "logprobs": logprobs
    }

    try:
        output = requests.post(API_URL, json=payload, headers=headers)
        output.raise_for_status()
        return output.json()
    except Exception as e:
        raise ValueError(f"Generation failed: {e}") from e


def parse_model_output(raw_output: dict) -> dict:
    try:
        content = raw_output["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Output parsing error ({e}): {raw_output}") from e
    
    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        output = m.group(1)
    else:
        output = content.strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Output parsing error ({e}): {output}") from e


def validate_evaluation(eval: dict, rubric: dict, metric: str) -> tuple[bool, list[str]]:
    errors = []

    SECTIONS = {
        "Summary Statement": list("ABCDEFG"),
        "Differential Diagnosis": list("ABCDEF"),
        "Explanation of Lead Diagnosis": list("ABC"),
        "Explanation of Alternative Diagnoses": list("ABC"),
        "Plan": list("ABCDE"),
    }

    for section_name in SECTIONS:
        if section_name not in eval:
            errors.append(f"Missing section: '{section_name}'")
            continue

        section = eval[section_name]
        expected_letters = SECTIONS[section_name]

        if "features" in section:
            for letter in expected_letters:
                if letter not in section["features"]:
                    errors.append(f"'{section_name}'.features missing feature '{letter}'")
                g = section["features"][letter]
                if not isinstance(g, bool):
                    errors.append(f"'{section_name}'.features['{letter}'] is not bool: {g}")

        if "confidence" in section:
            for letter in expected_letters:
                if letter not in section["confidence"]:
                    errors.append(f"'{section_name}'.confidence missing feature '{letter}'")
                else:
                    c = section["confidence"][letter]
                    if not isinstance(c, (int, float)):
                        errors.append(f"'{section_name}'.confidence['{letter}'] is not numerical: {c}")
                    elif metric == "vce" and not (0.5 <= float(c) <= 1.0):
                        errors.append(f"'{section_name}'.confidence['{letter}'] is out of range: {c}")

    return len(errors) == 0, errors


def save_results(results: list[dict], model_id: str, metric: str) -> str:
    id = model_id.replace("/", "_")
    path = f"results/{id}_{metric}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def extract_umsp(raw_output: dict) -> float:
    token_logprobs = [x["logprob"] for x in raw_output["choices"][0]["logprobs"]["content"]]
    # print(f"Logprobs: {token_logprobs}")
    return -sum(token_logprobs) # NLL


def normalize_umsp(umsp_values: list[float]) -> list[float]:
    q98 = float(np.percentile(umsp_values, 98))
    clipped = [min(u, q98) for u in umsp_values]
    min_u = min(clipped)
    denom = q98 - min_u
    if denom == 0:
        return [1.0] * len(umsp_values)
    return [1.0 - (u - min_u) / denom for u in clipped]


# -----------------------------------------------------------------------------------


def evaluate_vce(model_id: str):
    print(f"\n\nEvaluating VCE on {model_id}...")

    with open("dataset/sims.json") as f:
        sims = json.load(f)
    with open("dataset/rubric.json") as f:
        rubric = json.load(f)
    with open("prompts/evaluate_vce.txt") as f:
        base_prompt = f.read()

    results = []
    skipped = []

    for i, sim in enumerate(sims):
        print(f"[{i+1}/{len(sims)}]")

        evaluation = {}
        for k, v in rubric.items():
            system_prompt = build_system_prompt(base_prompt, v)
            user_prompt = ""
            for section in v['sections']:
                user_prompt += f"{section.upper()}:\n{sim['post_note_inputs'][section]}\n\n"
            
            try:
                raw_output = generate(model_id, system_prompt, user_prompt, logprobs=False)
                model_output = parse_model_output(raw_output)
                # print(model_output)
            except Exception as e:
                print(f"ERROR (API/parse): {e}")
                continue
            evaluation[k] = model_output

        is_valid, errors = validate_evaluation(evaluation, rubric, "vce")
        if not is_valid:
            print(f"ERROR (validation): {errors}")
            skipped.append(sim["_id"])
            continue

        doc = {
            "username": model_id,
            "sim_id": sim["_id"],
            "evaluation": evaluation
        }
        results.append(doc)

    print(f"\nCompleted: {len(results)}/{len(sims)} evaluations")
    if skipped:
        print(f"Skipped sims: {skipped}")

    path = save_results(results, model_id, "vce")
    print(f"Results saved to: {path}")


def evaluate_vce_rem(model_id: str, sim_ids: list[str]):
    print(f"\n\nEvaluating VCE remainders ({sim_ids}) on {model_id}...")

    with open("dataset/sims.json") as f:
        sims = json.load(f)
    with open("dataset/rubric.json") as f:
        rubric = json.load(f)
    with open("prompts/evaluate_vce.txt") as f:
        base_prompt = f.read()

    results = []
    skipped = []

    n = 0
    for i, sim in enumerate(sims):
        if sim["_id"] not in sim_ids:
            continue
        n += 1
        print(f"[{n}/{len(sim_ids)}]")

        evaluation = {}
        for k, v in rubric.items():
            system_prompt = build_system_prompt(base_prompt, v)
            user_prompt = ""
            for section in v['sections']:
                user_prompt += f"{section.upper()}:\n{sim['post_note_inputs'][section]}\n\n"
            
            try:
                raw_output = generate(model_id, system_prompt, user_prompt, logprobs=False)
                model_output = parse_model_output(raw_output)
                # print(model_output)
            except Exception as e:
                print(f"ERROR (API/parse): {e}")
                continue
            evaluation[k] = model_output

        is_valid, errors = validate_evaluation(evaluation, rubric, "vce")
        if not is_valid:
            print(f"ERROR (validation): {errors}")
            skipped.append(sim["_id"])
            continue

        doc = {
            "username": model_id,
            "sim_id": sim["_id"],
            "evaluation": evaluation
        }
        results.append(doc)

    print(f"\nCompleted: {len(results)}/{len(sim_ids)} evaluations")
    if skipped:
        print(f"Skipped sims: {skipped}")

    clean_id = model_id.replace("/", "_")
    path = f"results/{clean_id}_vce.json"
    with open(path, "r") as f:
        data = json.load(f)
    data.extend(results)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results added to: {path}")


# See section 3.2: https://arxiv.org/html/2510.20460v1
def evaluate_msp(model_id: str):
    print(f"\n\nEvaluating MSP on {model_id}...")

    with open("dataset/sims.json") as f:
        sims = json.load(f)
    with open("dataset/rubric.json") as f:
        rubric = json.load(f)
    with open("prompts/evaluate.txt") as f:
        base_prompt = f.read()

    # Pass 1: collect grades and raw UMSP values for every (sim, section)
    results = []
    skipped = []
    all_umsps = []

    for i, sim in enumerate(sims):
        print(f"[{i+1}/{len(sims)}]")

        evaluation = {}
        umsps = []
        for k, v in rubric.items():
            system_prompt = build_system_prompt(base_prompt, v)
            user_prompt = ""
            for section in v['sections']:
                user_prompt += f"{section.upper()}:\n{sim['post_note_inputs'][section]}\n\n"

            try:
                raw_output = generate(model_id, system_prompt, user_prompt, logprobs=True)
                # print(print(json.dumps(raw_output, indent=2)))
                # print(raw_output)
                model_output = parse_model_output(raw_output)
                umsp = extract_umsp(raw_output)
            except Exception as e:
                print(f"ERROR (API/parse): {e}")
                continue
            
            evaluation[k] = model_output
            umsps.append(umsp)

        is_valid, errors = validate_evaluation(evaluation, rubric, "msp")
        if not is_valid:
            print(f"ERROR (validation): {errors}")
            skipped.append(sim["_id"])
            continue
        
        doc = {
            "username": model_id,
            "sim_id": sim["_id"],
            "evaluation": evaluation
        }
        results.append(doc)
        all_umsps.extend(umsps)
        print(umsps)

    # Normalize to CMSPs
    cmsp_values = normalize_umsp(all_umsps)
    print(f"CMSPs: {cmsp_values}")

    # Add section-level confidence to results (same cmsp for whole section unfort)
    idx = 0
    for doc in results:
        for k, v in rubric.items():
            cmsp = cmsp_values[idx]
            idx += 1
            doc["evaluation"][k]["confidence"] = {letter: cmsp for letter in v["features"]}

    print(f"\nCompleted: {len(results)}/{len(sims)} evaluations")
    if skipped:
        print(f"Skipped sims: {skipped}")

    path = save_results(results, model_id, "msp")
    print(f"Results saved to: {path}")


def evaluate_sc(model_id: str):
    NUM_SAMPLES = 5
    print(f"\n\nEvaluating SC (N={NUM_SAMPLES}) on {model_id}...")

    with open("dataset/sims.json") as f:
        sims = json.load(f)
    with open("dataset/rubric.json") as f:
        rubric = json.load(f)
    with open("prompts/evaluate.txt") as f:
        base_prompt = f.read()

    results = []
    skipped = []

    for i, sim in enumerate(sims):
        print(f"[{i+1}/{len(sims)}]")

        evaluation = {}
        for k, v in rubric.items():
            system_prompt = build_system_prompt(base_prompt, v)
            user_prompt = ""
            for section in v["sections"]:
                user_prompt += f"{section.upper()}:\n{sim['post_note_inputs'][section]}\n\n"
            
            samples = {letter: [] for letter in v["features"]}
            for i in range(NUM_SAMPLES):
                try:
                    raw_output = generate(model_id, system_prompt, user_prompt, logprobs=False)
                    model_output = parse_model_output(raw_output)
                    # print(model_output)
                except Exception as e:
                    print(f"ERROR (API/parse): {e}")
                    continue
                for letter, grade in model_output["features"].items():
                    samples[letter].append(grade)    
            # print(samples)

            consensus = {"features": {}, "confidence": {}}
            for letter, votes in samples.items():
                maj = sum(votes) > (len(votes)/2)
                n_maj = sum(v == maj for v in votes)
                agreement = n_maj / len(votes)
                consensus["features"][letter] = maj
                consensus["confidence"][letter] = agreement

            evaluation[k] = consensus
            print(consensus)

        is_valid, errors = validate_evaluation(evaluation, rubric, "vce")
        if not is_valid:
            print(f"ERROR (validation): {errors}")
            skipped.append(sim["_id"])
            continue

        doc = {
            "username": model_id,
            "sim_id": sim["_id"],
            "evaluation": evaluation
        }
        results.append(doc)

    print(f"\nCompleted: {len(results)}/{len(sims)} evaluations")
    if skipped:
        print(f"Skipped sims: {skipped}")

    path = save_results(results, model_id, "sc")
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    # evaluate_vce("fireworks/kimi-k2p5")
    evaluate_vce_rem("fireworks/kimi-k2p5", ['68465ed820fd80066ac57d5c', '68465ed820fd80066ac57d58'])
