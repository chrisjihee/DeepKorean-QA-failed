import re


def horizontal_line(c="=", w=256, t=0, b=0):
    return "\n" * t + c * w + "\n" * b


morpheme_pattern = re.compile("([^ ]+?/[A-Z]{2,3})[+]?")


def to_morphemes(text: str, pattern=morpheme_pattern):
    return ' '.join(x.group(1) for x in pattern.finditer(text))
