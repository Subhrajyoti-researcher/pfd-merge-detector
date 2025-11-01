# src/vlm_client.py
import os
import json
import requests
from lxml import etree

# Configure these if you want different defaults
DEFAULT_MODEL = "gemini-pro-2.5"
GENAI_REST_ENDPOINT = "https://api.generativeai.googleapis.com/v1/models/{model}:generate"
API_KEY_ENV = "GEN_AI_API_KEY"

def _get_api_key():
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {API_KEY_ENV} not set. Set it to your Gemini API key.")
    return api_key

def call_gemini_pro(prompt: str,
                    model: str = DEFAULT_MODEL,
                    max_output_tokens: int = 2048,
                    temperature: float = 0.0,
                    top_p: float = 1.0,
                    timeout_seconds: int = 60):
    """
    Call Gemini Pro (Generative AI REST) with a plain-text prompt.
    Returns the raw text response from the model.
    """
    api_key = _get_api_key()
    url = GENAI_REST_ENDPOINT.format(model=model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # Payload shape: safe minimal. Some deployments expect different shapes; adjust if needed.
    payload = {
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if resp.status_code != 200:
        raise RuntimeError(f"Generative API error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Try several common shapes to extract text:
    # 1) { "output": [{ "content": "..." }, ...] }
    # 2) { "candidates": [{ "content": "..." }], "text": "..." }
    # 3) { "text": "..." }
    # fallback to raw json string if no text found.
    text = None
    if isinstance(data, dict):
        if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
            # some deployments put content under output[0].content
            first = data["output"][0]
            if isinstance(first, dict) and "content" in first:
                text = first["content"]
            elif isinstance(first, str):
                text = first
        if not text and "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"])>0:
            cand = data["candidates"][0]
            if isinstance(cand, dict) and "content" in cand:
                text = cand["content"]
            elif isinstance(cand, str):
                text = cand
        if not text and "text" in data and isinstance(data["text"], str):
            text = data["text"]

    if not text:
        # last resort: stringify the JSON for debugging
        text = json.dumps(data, indent=2)

    return text

def validate_and_save_xml(xml_text: str, out_path: str):
    """
    Validate xml_text is well-formed XML and save to out_path.
    If invalid, raise RuntimeError with a short preview.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
        xml_bytes = etree.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(xml_bytes)
        return out_path
    except Exception as e:
        # Save raw response for debugging
        raw_debug = out_path + ".raw.txt"
        with open(raw_debug, "w", encoding="utf-8") as fd:
            fd.write(xml_text)
        raise RuntimeError(f"Returned text is not valid XML: {e}. Raw response saved to {raw_debug}")

if __name__ == "__main__":
    # CLI convenience: read prompt from file, call model, save xml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default="data/output/ped1_vlm_prompt.txt")
    parser.add_argument("--out-xml", default="data/output/ped1_merges.xml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    print("Calling Gemini Pro model:", args.model)
    response_text = call_gemini_pro(prompt,
                                   model=args.model,
                                   max_output_tokens=args.max_tokens,
                                   temperature=0.0)
    print("Model call done; validating XML...")
    saved = validate_and_save_xml(response_text, args.out_xml)
    print("Saved validated XML to:", saved)