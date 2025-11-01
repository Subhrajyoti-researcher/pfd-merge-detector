# 🔬 PFD Merge Detector (Vision-Language Model)

This project automatically identifies **merge lines in Process Flow Diagrams (PFDs)** where two or more process streams **join with different temperature fluids**.  
Once merge lines are detected, their corresponding **line numbers are searched within the P&ID diagrams** and **highlighted visually** for verification.  
It leverages **Vision-Language Models (VLMs)** to understand the diagram structure, stream connections, and temperature annotations, producing structured **XML outputs** for further engineering analysis.

---

## 🎯 Objective

- Detect merge lines in **PFD images** where two or more streams converge with different temperature fluids.  
- Extract and list **merge line numbers** and associated **stream temperature data**.  
- Locate the same merge lines within **P&ID images** and **highlight them** for visualization.  
- Generate structured **XML output** containing merge metadata for downstream engineering or digital twin applications.

---

## 🧠 How It Works

1. **Input:**  
   - Process Flow Diagram (PFD) image  
   - Optional P&ID image for merge highlighting  

2. **OCR Extraction:**  
   - Text, line numbers, and temperature annotations are extracted using `pytesseract`.  

3. **Vision-Language Reasoning:**  
   - A VLM analyzes the image and OCR output to understand **stream topology**, **flow direction**, and **merge junctions**.  

4. **Merge Identification:**  
   - Lines where multiple streams join are analyzed for **temperature differences**.  
   - Merge coordinates and temperature data are extracted.  

5. **Cross-Reference with P&ID:**  
   - Detected merge line numbers are searched within P&ID images.  
   - The same lines are **highlighted** to visualize merge locations.  

6. **Output:**  
   - XML file containing merge IDs, coordinates, incoming/outgoing streams, and temperature differences.  

---

## 📄 Example Output (XML)

```xml
<merges>
  <merge>
    <merge_id>1</merge_id>
    <location_bbox>105,220,40,20</location_bbox>
    <incoming_streams>
      <stream id="A" text="330A" temp="215" unit="C"/>
      <stream id="B" text="339A" temp="144" unit="C"/>
    </incoming_streams>
    <outgoing_stream text="Debutanizer Feed"/>
    <temperature_difference>71</temperature_difference>
    <highlighted_in_pid>true</highlighted_in_pid>
  </merge>
</merges>
