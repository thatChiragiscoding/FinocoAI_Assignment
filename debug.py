# debug_pdf.py — run this once to see raw PDF text
import fitz

doc = fitz.open("paper/deep_research_survey.pdf")

# Print first 3 pages line by line with line numbers
for page_num in range(min(3, len(doc))):
    page = doc[page_num]
    text = page.get_text("text")
    print(f"\n{'='*60}")
    print(f"PAGE {page_num + 1}")
    print('='*60)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        print(f"{i:3d} | '{line}'")