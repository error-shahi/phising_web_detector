import re
from urllib.parse import urlparse

def extract_features(url):
    parsed = urlparse(url)

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('%'),
        url.count('/'),
        url.count('='),
        1 if parsed.scheme == 'https' else 0,
        1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        1 if '//' in parsed.path else 0,
        len(parsed.netloc),
        1 if parsed.netloc.count('.') > 3 else 0
    ]
