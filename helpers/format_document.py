def format_docs(retrieved_docs):
    formatted = []
    for doc in retrieved_docs:
        if hasattr(doc, "page_content"):   # Document object
            formatted.append(doc.page_content)
        else:   # plain string
            formatted.append(str(doc))
    return "\n\n".join(formatted)
