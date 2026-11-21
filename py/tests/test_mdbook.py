from intgrad.corpus.mdbook import classify, stats

PARA = ("The quick brown fox considers the river, then walks along the bank "
        "until the bridge appears, and crosses it before the evening light fades.")


def wrap(text, width=60):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(out + [line])


def test_hard_wrapped_prose_counts_as_prose():
    doc = ("# Chapter\n\n" + "\n\n".join(wrap(PARA) for _ in range(20))).encode()
    s = stats(doc)
    assert s["prose"] > 0.9 and s["wrapped"] > 0.9
    assert classify(s) == "keep"


def test_table_heavy_book_is_poor():
    rows = "\n".join(f"| {i} | {i * i} | value |" for i in range(400))
    doc = (PARA + "\n\n" + rows).encode()
    assert classify(stats(doc)) == "poor"


def test_code_and_math_are_content():
    code = "```python\n" + "\n".join(f"x{i} = {i} * 2" for i in range(200)) + "\n```\n"
    math = "$$\n\\frac{a}{b} = \\sum_i \\alpha_i \\beta_i\n$$\n"
    doc = (PARA + "\n\n" + code + "\n" + math * 20).encode()
    s = stats(doc)
    assert s["prose"] < 0.25 and s["code"] > 0.5
    assert classify(s) != "poor"


def test_language_tag_is_metadata():
    de = ("Der Fluss ist nicht weit, und die Brücke ist auch nicht alt, aber sie "
          "ist mit dem Weg verbunden, der von dem Dorf kommt. ") * 5
    assert stats(de.encode())["lang"] == "de"
    assert stats((PARA * 5).encode())["lang"] == "en"
