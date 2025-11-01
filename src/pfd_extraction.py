"""
Colab-friendly script: Identify PFD lines that "merge" with different-temperature fluids using Gemini Pro.

Workflow:
1. Upload one or more PFD images in Colab.
2. Script runs OCR (pytesseract) to extract text labels and coordinates (temperatures, stream names).
3. Builds a compact prompt including extracted text and coordinates and asks Gemini Pro to identify "merge" lines where two streams with different temperatures join.
4. Receives LLM response and writes a structured XML document describing the merge lines.

Requirements (run in Colab):
!pip install --upgrade google-generativeai pillow pytesseract lxml
!apt-get update && apt-get install -y tesseract-ocr

Before running set your Gemini/Generative AI API key in an environment variable in Colab (safer) like:
import os
os.environ['GEN_AI_API_KEY'] = 'YOUR_API_KEY'

This script uses the unofficial-but-common `google.generativeai` Python client pattern. If your organization uses Vertex AI or a different client, substitute the client call in call_gemini().

Note: adjust model name in call_gemini() if your account uses a different model id for Gemini Pro.
"""

# Colab-ready code
import os
import re
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import pytesseract
import requests
from lxml import etree

# --------------------
# User config
# --------------------
# Put your Generative AI API key in the GEN_AI_API_KEY environment variable in Colab.
GEN_AI_API_KEY = os.environ.get('GEN_AI_API_KEY')
# Model name (edit if needed)
MODEL_NAME = "gemini-pro"  # adjust to your tenant/model id if required
# Generative AI REST endpoint template (this is a common pattern; adjust if your org requires Vertex AI calls)
GENAI_REST_ENDPOINT = "https://api.generativeai.googleapis.com/v1/models/{model}:generate"

# --------------------
# Helpers: OCR and data extraction
# --------------------

def extract_text_blocks(image_path):
    """Use pytesseract to extract text blocks with bounding boxes.
    Returns list of dict: {text, left, top, width, height, conf}
    """
    img = Image.open(image_path).convert('RGB')
    # pytesseract image_to_data
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    blocks = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1
        block = {
            'text': text,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'conf': conf
        }
        blocks.append(block)
    return blocks


def find_temperature_terms(blocks):
    """From OCR blocks, find tokens that look like temperatures or stream labels.
    Return a list of entries with numeric temperature if present.
    """
    temp_pattern = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*(?:°|deg|degrees)?\s*(?:C|c|F|f)?")
    entries = []
    for b in blocks:
        m = temp_pattern.search(b['text'])
        if m:
            temp_val = float(m.group(1))
            unit = 'C' if re.search(r"[°\s](?:C|c)\b", b['text']) else None
            entries.append({
                'text': b['text'],
                'temp': temp_val,
                'unit': unit,
                'left': b['left'],
                'top': b['top'],
                'width': b['width'],
                'height': b['height'],
                'conf': b['conf']
            })
    # If no explicit temperature tokens, still keep named streams for context
    return entries

# --------------------
# Build the prompt to send to Gemini Pro
# --------------------

def build_prompt(image_filename, ocr_blocks, temp_entries, max_blocks=60):
    """Construct a compact prompt summarizing OCR findings and asking the model to output XML describing merge lines.
    We provide coordinates to help the model reason about proximity.
    """
    # Keep prompt short but informative
    header = (
        "You are an expert chemical process diagram analyst.\n"
        "I will provide OCR extracted text items from a Process Flow Diagram (PFD) image.\n"
        "Identify lines (pipes/streams) where two or more streams "
        "merge and the merging streams have clearly different temperatures.\n"
        "For each merge, output an XML element <merge> with child elements:"
        "<merge_id>, <location_bbox> (left,top,width,height), <incoming_streams> (each with id, text, temp, unit, bbox), "
        "<outgoing_stream> (id, text, temp, unit, bbox if identifiable), and <temperature_difference>.\n"
        "If a stream has no explicit temperature, leave temp/unit empty but include text and bbox.\n"
        "Use degrees Celsius if a unit is present; otherwise leave unit blank.\n"
        "Return ONLY valid XML. Do not add any commentary.\n\n"
    )

    # Compose OCR summary (limit length)
    items = []
    for i, b in enumerate(ocr_blocks[:max_blocks]):
        items.append(f"id={i}|text={b['text']}|left={b['left']}|top={b['top']}|w={b['width']}|h={b['height']}|conf={b['conf']}")
    ocr_summary = "\n".join(items)

    temp_summary = []
    for i, t in enumerate(temp_entries):
        temp_summary.append(f"t{i}|text={t['text']}|temp={t.get('temp','','')}|unit={t.get('unit','')}|left={t['left']}|top={t['top']}")
    temp_summary = "\n".join(temp_summary)

    prompt = (
        header
        + f"Image filename: {image_filename}\n"
        + "OCR items (id|text|left|top|w|h|conf):\n"
        + ocr_summary
        + "\n\nTemperature-like items (id|text|temp|unit|left|top):\n"
        + temp_summary
        + "\n\nNow produce XML describing each merge where two or more streams join and have different temperatures. "
        + "Compute temperature_difference as absolute difference between numeric temps when possible.\n"
    )
    return prompt

# --------------------
# Call Gemini Pro / Generative AI
# --------------------

def call_gemini(prompt, api_key=GEN_AI_API_KEY, model=MODEL_NAME):
    """Call the Generative AI REST endpoint. Returns text output from the model.
    NOTE: You may need to adapt this to your org's Vertex AI or other setup.
    """
    if not api_key:
        raise RuntimeError("GEN_AI_API_KEY not set. Please set environment variable in Colab before running.")

    url = GENAI_REST_ENDPOINT.format(model=model)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'prompt': prompt,
        'max_output_tokens': 1024,
        'temperature': 0.0
    }
    # Some deployments use a slightly different request shape; adapt if necessary.
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Generative API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Extract text in a few common shapes
    text = None
    if 'output' in data:
        # e.g., {"output": [{"content": "..."}]}
        if isinstance(data['output'], list) and len(data['output'])>0 and 'content' in data['output'][0]:
            text = data['output'][0]['content']
    if not text:
        # fallback
        text = data.get('text') or data.get('response') or json.dumps(data)
    return text

# --------------------
# XML saving
# --------------------

def validate_and_save_xml(xml_text, out_path):
    """Attempt to parse and save XML. If invalid, raise an error with a helpful message.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_text.encode('utf-8'), parser=parser)
        # pretty print
        xml_bytes = etree.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True)
        with open(out_path, 'wb') as f:
            f.write(xml_bytes)
        return out_path
    except Exception as e:
        raise RuntimeError(f"Returned text is not valid XML: {e}\n---Returned text---\n{xml_text[:4000]}")

# --------------------
# High-level pipeline for a single image
# --------------------

def process_image_file(image_path, out_xml_path):
    print(f"Processing {image_path}...")

    ocr_blocks = extract_text_blocks(image_path)
    temp_entries = find_temperature_terms(ocr_blocks)

    prompt = build_prompt(os.path.basename(image_path), ocr_blocks, temp_entries)

    print("Sending prompt to Gemini Pro (generative API)...")
    response_text = call_gemini(prompt)

    print("Model response received. Validating and saving XML...")
    saved_path = validate_and_save_xml(response_text, out_xml_path)
    print(f"Saved XML to: {saved_path}")
    return saved_path

# --------------------
# Example usage in Colab
# --------------------
if __name__ == '__main__':
    # In Colab, use the file upload widget to upload images
    # from google.colab import files
    # uploaded = files.upload()
    # for filename in uploaded.keys():
    #     process_image_file(filename, filename + '.merges.xml')

    # For local testing, set a path
    sample_image = 'pfd_sample.jpg'  # replace with uploaded file name
    if os.path.exists(sample_image):
        try:
            out = process_image_file(sample_image, sample_image + '.merges.xml')
        except Exception as ex:
            print('Error:', ex)
    else:
        print('\nThis script is ready. To use in Colab:')
        print('1) Upload your PFD images with:')
        print("   from google.colab import files\n   uploaded = files.upload()\n   for fn in uploaded: process_image_file(fn, fn + '.merges.xml')")
        print('2) Set your GEN_AI_API_KEY env var in Colab before calling the script:')
        print("   import os\n   os.environ['GEN_AI_API_KEY'] = 'YOUR_API_KEY'")
        print('3) Install dependencies:\n   !pip install --upgrade google-generativeai pillow pytesseract lxml\n   !apt-get update && apt-get install -y tesseract-ocr')

    print('\nDone.\n')
