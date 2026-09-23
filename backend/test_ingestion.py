from ingestion import load_and_split_pdf


PDF_PATH = r"G:\enterprise-AI-knowledge-assistant\data\sample_company_handbook.pdf"

chunks = load_and_split_pdf(PDF_PATH)

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)