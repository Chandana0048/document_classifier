from text_extractor import *

print("PDF Text:\n", extract_text_from_pdf("uploads/sample.pdf"))
print("DOCX Text:\n", extract_text_from_docx("uploads/sample.docx"))
print("TXT Text:\n", extract_text_from_txt("uploads/sample.txt"))
