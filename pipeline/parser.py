import pdfplumber
import re


def extract_text_from_pdf(file_path):
    """
    Extract full text from PDF
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_chapter_pages(text):
    """
    Extract chapter → page mapping from Table of Contents
    Example:
    CHAPTER 4 .... 35 → {4: 35}
    """
    pattern = r"CHAPTER\s+(\d+).*?(\d+)"
    matches = re.findall(pattern, text)

    chapter_pages = {}
    for chapter, page in matches:
        chapter_pages[int(chapter)] = int(page)

    return chapter_pages

def extract_chapter_4_from_pages(file_path, start_page, end_page):
    """
    Extract Chapter 4 using page numbers (best method)
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        start_idx = max(start_page - 1, 0)
        end_idx = min(end_page - 1, total_pages)

        for i in range(start_idx, end_idx):
            page = pdf.pages[i]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text

def extract_chapter_4(file_path):
    """
    Main function:
    1. Try page-based extraction (from TOC)
    2. Fallback to text-based extraction
    """
    full_text = extract_text_from_pdf(file_path)
    chapter_pages = extract_chapter_pages(full_text)
    start_page = chapter_pages[4]
    end_page = chapter_pages.get(5, start_page + 50)
    print(f"Using page-based extraction: {start_page} → {end_page}")
    return extract_chapter_4_from_pages(file_path, start_page, end_page)
