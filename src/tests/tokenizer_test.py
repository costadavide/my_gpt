from src.tokenizer import Tokenizer

def test_pre_tokenize():
    tokenizer = Tokenizer(vocab_size=1000)

    tests = [
        # 1. Basic words
        ("Hello world", ["Hello", " world"]),
        ("Hello World", ["Hello", " World"]),
        ("HELLO world", ["HELLO", " world"]),
        ("Hello", ["Hello"]),
        ("", []),

        # 2. Numbers
        ("123", ["123"]),
        ("Hello123", ["Hello", "123"]),
        ("123Hello", ["123", "Hello"]),
        ("I have 123 apples", ["I", " have", " 123", " apples"]),
        ("123.456", ["123", ".", "456"]),

        # 3. Punctuation and special characters
        ("Hello!", ["Hello", "!"]),
        ("Hello!!!", ["Hello", "!!!"]),
        ("Hello, world!", ["Hello", ",", " world", "!"]),
        ("Hello_world", ["Hello", "_", "world"]),
        ("a@b.com", ["a", "@", "b", ".", "com"]),
        ("Hello... How?", ["Hello", "...", " How", "?"]),

        # 4. Contractions
        ("don't", ["don", "'t"]),
        ("I'm happy", ["I", "'m", " happy"]),
        ("we're here", ["we", "'re", " here"]),
        ("they've gone", ["they", "'ve", " gone"]),
        ("she'll come", ["she", "'ll", " come"]),
        ("I'd like that", ["I", "'d", " like", " that"]),
        ("it's fine", ["it", "'s", " fine"]),

        # 5. Multiple spaces
        ("Hello  world", ["Hello", " ", " world"]),
        ("Hello   world", ["Hello", "  ", " world"]),
        ("Hello    world", ["Hello", "   ", " world"]),
        ("Hello   ", ["Hello", "   "]),
        ("   Hello", ["  ", " Hello"]),
        ("   ", ["   "]),

        # 6. Newlines and tabs
        ("Hello\nworld", ["Hello", "\n", "world"]),
        ("Hello\n\nworld", ["Hello", "\n", "\n", "world"]),
        ("Hello\tworld", ["Hello", "\t", "world"]),
        ("Hello \nworld", ["Hello", " ", "\n", "world"]),
        ("Hello\n world", ["Hello", "\n", " world"]),

        # 7. Mixed cases
        ("Hello, how are  you?",
         ["Hello", ",", " how", " are", " ", " you", "?"]),

        ("GPT-2 is awesome!",
         ["GPT", "-", "2", " is", " awesome", "!"]),

        ("Price: $123.45!",
         ["Price", ":", " $", "123", ".", "45", "!"]),

        # 8. Non-ASCII characters (treated as symbols in your regex)
        ("café", ["caf", "é"]),
        ("hello😊", ["hello", "😊"]),
    ]

    for i, (text, expected) in enumerate(tests):
        actual = tokenizer.pre_tokenize(text)

        assert actual == expected, (
            f"Test {i + 1} failed!\n"
            f"Input:    {text!r}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )

        assert "".join(actual) == text, (
            f"Test {i + 1}: Original text was not preserved!"
        )

    print(f"All {len(tests)} pre-tokenization tests passed!")

if __name__ == "__main__":
    test_pre_tokenize()