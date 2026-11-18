from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/tokenizer.json")

tests = [
    "347 + 285 = 632",
    "(300 + 200) + (40 + 80) + (7 + 5)",
    "Question: What is 12 * 7?\nAnswer: 84",
    "12345",
]
for text in tests:
    enc = tok.encode(text)
    print(enc.tokens)
    print("round trip ok:", tok.decode(enc.ids) == text)
    print()

single = sum(1 for n in range(1000) if len(tok.encode(str(n)).ids) == 1)
print(single, "of the numbers 0-999 are a single token")