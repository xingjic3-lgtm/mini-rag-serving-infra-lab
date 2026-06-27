from rag_pipeline import DEFAULT_DOCUMENT, DEFAULT_QUESTION, run_rag


def main():
    result = run_rag(DEFAULT_DOCUMENT, DEFAULT_QUESTION)

    print("[Document Loaded]")
    print(result.document)

    print("\n[Question]")
    print(result.question)

    print("\n[Retrieved Context]")
    print(result.retrieved_context)

    print("\n[Prompt]")
    print(result.prompt)

    print("\n[LLM Answer]")
    print(result.answer)


if __name__ == "__main__":
    main()
