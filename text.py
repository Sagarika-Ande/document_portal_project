# Test code for document ingestion and analysis using a PDFHandler and DocumentAnalyzer

import os
from pathlib import Path
from src.document_analyzer.data_ingestion import DocumentHandler       # Your PDFHandler class
from src.document_analyzer.data_analysis import DocumentAnalyzer  # Your DocumentAnalyzer class

# # Path to the PDF you want to test
PDF_PATH = r"D:\Users\asset\document_portal_project\data\document_analysis\NIPS-2017-attention-is-all-you-need-Paper.pdf"

# # Dummy file wrapper to simulate uploaded file (Streamlit style)
class DummyFile:
        def __init__(self, file_path):
            self.name = Path(file_path).name
            self._file_path = file_path

        def getbuffer(self):
         return open(self._file_path, "rb").read()
def main():
     try:
                   # ---------- STEP 1: DATA INGESTION ----------
         print("Starting PDF ingestion...")
         dummy_pdf = DummyFile(PDF_PATH)

         handler = DocumentHandler(session_id="test_ingestion_analysis")
        
         saved_path = handler.save_pdf(dummy_pdf)
         print(f"PDF saved at: {saved_path}")

         text_content = handler.read_pdf(saved_path)
         print(f"Extracted text length: {len(text_content)} chars\n")

         # ---------- STEP 2: DATA ANALYSIS ----------
         print("Starting metadata analysis...")
         analyzer = DocumentAnalyzer()  # Loads LLM + parser
        
         analysis_result = analyzer.analyze_document(text_content)

#         # ---------- STEP 3: DISPLAY RESULTS ----------
         print("\n=== METADATA ANALYSIS RESULT ===")
         for key, value in analysis_result.items():
             
             print(f"{key}: {value}")

     except Exception as e:
         print(f"Test failed: {e}")

if __name__ == "__main__":
    main()