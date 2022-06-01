import re
def _remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "<URL>", text)


def _remove_escaped_characters(text: str):
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    # When the scaped character is just before a ".", removing the escaped char will remove the
    # space between the dot and the new word. This is a quick workaround.
    return text.translate(translator).replace(".", ". ").replace(".  ", ". ")

def clean_text(text, remove_url=True, remove_escaped_chars=True):
    if remove_url:
        text=_remove_URL(text)
    if remove_escaped_chars:
        text=_remove_escaped_characters(text)
    return text