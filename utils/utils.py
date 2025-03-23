import tiktoken

def calculate_token_count(text, model="gpt-4"):
    """
    Calculate the number of tokens in a given text using the specified model.

    :param text: The text to calculate tokens for.
    :param model: The OpenAI model name (default is gpt-4).
    :return: The number of tokens.
    """
    try:
        # Load encoding based on the model
        encoding = tiktoken.encoding_for_model(model)
        # Encode the text to tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        raise ValueError(f"Error calculating tokens: {e}")
