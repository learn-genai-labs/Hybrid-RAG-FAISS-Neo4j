from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> list[Document]:
    """
    Load a PDF file and convert each page containing text
    into a LangChain Document.

    Each Document contains:
        page_content:
            Extracted text from the PDF page.

        metadata:
            source    - Path of the source PDF.
            file_name - Name of the PDF file.
            page      - Original PDF page number.
    """

    file_path = Path(pdf_path)

    # ---------------------------------------------------------
    # Validate input file
    # ---------------------------------------------------------

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Only PDF files are supported: {file_path.name}"
        )

    # ---------------------------------------------------------
    # Open PDF
    # ---------------------------------------------------------

    documents: list[Document] = []

    pdf = pymupdf.open(file_path)

    try:

        # -----------------------------------------------------
        # Process each PDF page
        # -----------------------------------------------------

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            # Skip pages that contain no extractable text
            if not text:
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "page": page_number,
                },
            )

            documents.append(document)

    finally:

        # -----------------------------------------------------
        # Always close PDF
        # -----------------------------------------------------

        pdf.close()

    return documents


# -------------------------------------------------------------
# Local test
# -------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "\nUsage:\n"
            "python -m backend.ingestion.pdf_loader "
            "\"data\\uploads\\your_file.pdf\"\n"
        )

        raise SystemExit(1)

    pdf_file = sys.argv[1]

    print("\nLoading PDF...")
    print(f"File: {pdf_file}")

    try:

        loaded_documents = load_pdf(pdf_file)

        print("\nPDF loaded successfully.")
        print(
            f"Pages with extracted text: "
            f"{len(loaded_documents)}"
        )

        if not loaded_documents:
            print(
                "\nNo extractable text was found in the PDF."
            )

            raise SystemExit(0)

        # -----------------------------------------------------
        # Display first page for verification
        # -----------------------------------------------------

        first_document = loaded_documents[0]

        print("\nFirst page metadata:")
        print(first_document.metadata)

        print("\nFirst page text preview:")
        print("-" * 60)

        print(
            first_document.page_content[:1000]
        )

        print("-" * 60)

    except Exception as error:

        print("\nPDF loading failed.")
        print(f"Error: {error}")

        raise