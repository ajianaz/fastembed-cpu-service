import tiktoken

def calculate_token_count(text, model="gpt-4"):
    """
    Calculate the number of tokens in a given text using the specified model.

    :param text: The text to calculate tokens for.
    :param model: The OpenAI model name (default is gpt-4).
    :return: The number of tokens.
    """
    try:
        # Jika model adalah OpenAI model, gunakan tiktoken
        if model.startswith("gpt-"):
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)

        # Jika bukan OpenAI model, gunakan pendekatan sederhana
        avg_chars_per_token = 4  # Asumsi rata-rata 4 karakter per token
        return len(text) // avg_chars_per_token
    except Exception as e:
        raise ValueError(f"Error calculating tokens: {e}")
