# src/gpt5_client.py
import os, json
from lxml import etree
from openai import OpenAI

def call_gpt5(prompt: str,
              model: str = "gpt-5",
              max_output_tokens: int = 2048,
              temperature: float = 0.0):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are an expert chemical process diagram analyst. Output valid XML only."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content.strip()

def validate_and_save_xml(xml_text: str, out_path: str):
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    xml_bytes = etree.tostring(root, pretty_print=True,
                               encoding="utf-8", xml_declaration=True)
    with open(out_path, "wb") as f:
        f.write(xml_bytes)
    return out_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default="data/output/ped1_vlm_prompt.txt")
    parser.add_argument("--out-xml", default="data/output/ped1_gpt5_merges.xml")
    args = parser.parse_args()

    with open(args.prompt_file) as f:
        prompt = f.read()

    print("Calling GPT-5…")
    xml_text = call_gpt5(prompt)
    saved = validate_and_save_xml(xml_text, args.out_xml)
    print("Saved XML to:", saved)