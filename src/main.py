from indexing import indexing
from generation import generate
from dotenv import find_dotenv, load_dotenv
import os


# https://python.langchain.com/docs/use_cases/question_answering/

def main():
    load_dotenv(find_dotenv())
    pdf_filename = 'California-Tenants-Guide.pdf'  # assuming in /test_files
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'test_files', pdf_filename)
    prompt = "What are my CA renter rights?"
    k = 6
    generate(pdf_path, prompt, k)
    """
    example response:
    Your California renter rights include limits on security deposit amounts, protection against unlawful discrimination,
    limits on rent increases, the right to withhold rent under certain circumstances, and the right to repair defects in 
    the rental unit. These rights are always present, regardless of what the rental agreement states. Thanks for asking!
    """

main()