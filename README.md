# 🔬 PFD Merge Detector (Vision-Language Model)

This project automatically identifies **merge lines in Process Flow Diagrams (PFDs)** where two or more process streams **join with different temperature fluids**.  
It leverages **Vision-Language Models (VLMs)** to understand the diagram structure, stream connections, and temperature annotations, producing structured **XML outputs** for further engineering analysis.

---

## 🧠 Overview

### 🎯 Objective
Detect merge locations in PFD images where two or more fluid streams combine at **different temperatures**.  
Each merge is represented as a structured `<merge>` XML element containing:
- Stream IDs and temperatures  
- Merging coordinates (bounding boxes)  
- Calculated temperature difference  

---

## 🏗️ Architecture

1. **Input:** PFD image (`.jpg`, `.png`, `.pdf`)
2. **OCR:** Extracts text, stream tags, and temperature data using `pytesseract`
3. **VLM Analysis:** A Vision-Language Model interprets the image and text contextually  
4. **Reasoning Layer:** Identifies merge points and calculates temperature differences  
5. **Output:** XML file describing each merge event and its characteristics  

---

## 📂 Repository Structure
