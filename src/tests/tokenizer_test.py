from src.tokenizer import Tokenizer
import pytest

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


def test_merge_pair():
    tokenizer = Tokenizer(vocab_size=300)

    assert tokenizer.merge_pair([1, 2, 3], (1, 2), 256) == [256, 3]
    assert tokenizer.merge_pair([1, 2, 1, 2], (1, 2), 256) == [256, 256]
    assert tokenizer.merge_pair([1, 1, 1, 1], (1, 1), 256) == [256, 256]
    assert tokenizer.merge_pair([1, 1, 1], (1, 1), 256) == [256, 1]
    assert tokenizer.merge_pair([1, 2, 3], (2, 3), 256) == [1, 256]
    assert tokenizer.merge_pair([1, 2, 3], (4, 5), 256) == [1, 2, 3]
    assert tokenizer.merge_pair([1], (1, 2), 256) == [1]
    assert tokenizer.merge_pair([], (1, 2), 256) == []



def test_train():
    # 1. Test initial vocabulary
    tokenizer = Tokenizer(vocab_size=260)

    assert len(tokenizer.vocab) == 256
    assert len(tokenizer.merges) == 0
    assert tokenizer.vocab[0] == b"\x00"
    assert tokenizer.vocab[255] == b"\xff"
    print("Initial vocabulary test passed")

    # 2. Test a single merge
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("ababab")

    assert len(tokenizer.vocab) == 257
    assert tokenizer.vocab[256] == b"ab"
    assert tokenizer.merges == {(97, 98): 256}
    print("Single merge test passed")

    # 3. Test multiple successive merges
    tokenizer = Tokenizer(vocab_size=258)
    tokenizer.train("ababab")

    assert len(tokenizer.vocab) == 258
    assert tokenizer.vocab[256] == b"ab"
    assert tokenizer.vocab[257] == b"abab"
    assert tokenizer.merges[(97, 98)] == 256
    assert tokenizer.merges[(256, 256)] == 257
    print("Multiple merges test passed")

    # 4. Test that merges respect pre-token boundaries
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("a b")

    # Pre-tokenization: ["a", " b"]
    # The pair (a, space) must never be merged.
    assert tokenizer.vocab[256] == b" b"
    assert (97, 32) not in tokenizer.merges
    print("Pre-token boundary test passed")

    # 5. Test vocabulary size limit
    tokenizer = Tokenizer(vocab_size=260)
    tokenizer.train("abababababababab")

    assert len(tokenizer.vocab) == 260
    assert len(tokenizer.merges) == 4
    print("Vocabulary size limit test passed")

    # 6. Test stopping when no pairs remain
    tokenizer = Tokenizer(vocab_size=300)
    tokenizer.train("a")

    assert len(tokenizer.vocab) == 256
    assert len(tokenizer.merges) == 0
    print("No available pairs test passed")

    # 7. Test empty corpus
    tokenizer = Tokenizer(vocab_size=300)
    tokenizer.train("")

    assert len(tokenizer.vocab) == 256
    assert len(tokenizer.merges) == 0
    print("Empty corpus test passed")

    # 8. Test UTF-8 characters
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("ééé")

    assert len(tokenizer.vocab) == 257
    assert tokenizer.vocab[256] == b"\xc3\xa9"
    assert tokenizer.merges[(195, 169)] == 256
    print("UTF-8 test passed")

    # 9. Test that every learned token is a valid
    # concatenation of its component tokens
    tokenizer = Tokenizer(vocab_size=270)
    tokenizer.train("hello hello hello world world")

    for (left, right), new_id in tokenizer.merges.items():
        assert tokenizer.vocab[new_id] == (
            tokenizer.vocab[left] + tokenizer.vocab[right]
        )

    print("Vocabulary consistency test passed")

    print("\nAll BPE training tests passed!")


def test_encode():
    # 1. Encoding without learned merges
    tokenizer = Tokenizer(vocab_size=256)

    assert tokenizer.encode("hello") == [104, 101, 108, 108, 111]
    assert tokenizer.encode("a") == [97]
    print("Encoding without merges test passed")

    # 2. Encoding with a single learned merge
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("ababab")

    assert tokenizer.encode("ab") == [256]
    assert tokenizer.encode("abab") == [256, 256]
    assert tokenizer.encode("abc") == [256, 99]
    print("Single merge test passed")

    # 3. Encoding with multiple successive merges
    tokenizer = Tokenizer(vocab_size=258)
    tokenizer.train("ababab")

    assert tokenizer.encode("abab") == [257]
    assert tokenizer.encode("ababab") == [257, 256]
    print("Multiple successive merges test passed")

    # 4. Verify merge priority
    tokenizer = Tokenizer(vocab_size=258)

    tokenizer.merges = {
        (97, 98): 256,   # a + b -> ab
        (98, 99): 257    # b + c -> bc
    }
    tokenizer.vocab[256] = b"ab"
    tokenizer.vocab[257] = b"bc"

    # Both pairs are available, but (a,b) has higher priority
    assert tokenizer.encode("abc") == [256, 99]
    print("Merge priority test passed")

    # 5. Verify merges never cross pre-token boundaries
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("a b")

    assert tokenizer.encode("a b") == [97, 256]
    assert tokenizer.encode("ab") == [97, 98]
    print("Pre-token boundary test passed")

    # 6. Encoding unknown words
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("ababab")

    # Unseen text must still be encodable
    assert tokenizer.encode("xyz") == [120, 121, 122]
    print("Unknown words test passed")

    # 7. Empty string
    tokenizer = Tokenizer(vocab_size=260)
    tokenizer.train("hello")

    assert tokenizer.encode("") == []
    print("Empty string test passed")

    # 8. Unicode without learned merges
    tokenizer = Tokenizer(vocab_size=256)

    assert tokenizer.encode("é") == [195, 169]
    assert tokenizer.encode("€") == [226, 130, 172]
    assert tokenizer.encode("😊") == list("😊".encode("utf-8"))
    print("Unicode without merges test passed")

    # 9. Unicode with learned merges
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("ééé")

    assert tokenizer.encode("é") == [256]
    assert tokenizer.encode("éé") == [256, 256]
    print("Unicode with merges test passed")

    # 10. Repeated characters
    tokenizer = Tokenizer(vocab_size=258)
    tokenizer.train("aaaaaa")

    assert tokenizer.encode("aa") == [256]
    assert tokenizer.encode("aaaa") == [257]
    assert tokenizer.encode("aaaaaa") == [257, 256]
    print("Repeated characters test passed")

    # 11. All generated IDs must exist in the vocabulary
    tokenizer = Tokenizer(vocab_size=280)
    tokenizer.train("hello world hello world")

    texts = [
        "hello",
        "world",
        "hello world",
        "unknown text",
        "Hello, world!",
        "12345",
        "é😊",
        ""
    ]

    for text in texts:
        encoded = tokenizer.encode(text)
        assert all(token_id in tokenizer.vocab for token_id in encoded)

    print("Token ID validity test passed")

    print("\nAll encode tests passed!")


def test_encode_against_reference():
    tokenizer = Tokenizer(vocab_size=300)
    tokenizer.train(
        "hello world hello world "
        "testing tokenizer testing tokenizer "
        "abababab aaaaaa"
    )

    def reference_encode(text):
        result = []

        for word in tokenizer.pre_tokenize(text):
            tokens = list(word.encode("utf-8"))

            for pair, new_id in tokenizer.merges.items():
                tokens = tokenizer.merge_pair(tokens, pair, new_id)

            result.extend(tokens)

        return result

    texts = [
        "hello world",
        "testing tokenizer",
        "abababab",
        "aaaaaa",
        "unseen words",
        "Hello, world!",
        "123456",
        "é😊",
        "",
    ]

    for text in texts:
        actual = tokenizer.encode(text)
        expected = reference_encode(text)

        assert actual == expected, (
            f"Encoding mismatch for {text!r}: "
            f"expected {expected}, got {actual}"
        )

    print("Reference implementation test passed")


def test_decode():
    # 1. Basic ASCII decoding
    tokenizer = Tokenizer(vocab_size=256)

    assert tokenizer.decode([104, 101, 108, 108, 111]) == "hello"
    assert tokenizer.decode([97]) == "a"
    assert tokenizer.decode([72, 105, 33]) == "Hi!"
    print("Basic ASCII decoding test passed")

    # 2. Empty token sequence
    assert tokenizer.decode([]) == ""
    print("Empty sequence test passed")

    # 3. Spaces, tabs, and newlines
    assert tokenizer.decode([104, 105, 32, 116, 104, 101, 114, 101]) == "hi there"
    assert tokenizer.decode([97, 10, 98]) == "a\nb"
    assert tokenizer.decode([97, 9, 98]) == "a\tb"
    assert tokenizer.decode([32, 32, 97, 32, 32]) == "  a  "
    print("Whitespace decoding test passed")

    # 4. Unicode characters
    assert tokenizer.decode([195, 169]) == "é"
    assert tokenizer.decode([226, 130, 172]) == "€"
    assert tokenizer.decode(list("😊".encode("utf-8"))) == "😊"
    print("Unicode decoding test passed")

    # 5. Decoding a single learned BPE token
    tokenizer = Tokenizer(vocab_size=257)
    tokenizer.train("ababab")

    assert tokenizer.decode([256]) == "ab"
    assert tokenizer.decode([256, 256]) == "abab"
    print("Single BPE merge decoding test passed")

    # 6. Decoding multiple learned BPE tokens
    tokenizer = Tokenizer(vocab_size=258)
    tokenizer.train("ababab")

    assert tokenizer.decode([257]) == "abab"
    assert tokenizer.decode([257, 256]) == "ababab"
    print("Multiple BPE merges decoding test passed")

    # 7. Mixed original bytes and learned BPE tokens
    assert tokenizer.decode([256, 99, 100]) == "abcd"
    assert tokenizer.decode([97, 256, 98]) == "aabb"
    print("Mixed token decoding test passed")

    # 8. Unicode split across multiple tokens
    tokenizer = Tokenizer(vocab_size=256)

    # UTF-8 characters must be reconstructed before decoding
    assert tokenizer.decode([195, 169, 195, 169]) == "éé"
    assert tokenizer.decode(list("你好".encode("utf-8"))) == "你好"
    print("Split Unicode decoding test passed")

    # 9. Invalid token IDs
    with pytest.raises(KeyError):
        tokenizer.decode([999])

    with pytest.raises(KeyError):
        tokenizer.decode([-1])

    print("Invalid token ID test passed")

    # 10. Invalid UTF-8 byte sequences
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([195])

    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([255])

    print("Invalid UTF-8 test passed")

    # 11. Encode-decode round trip
    tokenizer = Tokenizer(vocab_size=300)
    tokenizer.train(
        "hello world hello world "
        "testing tokenizer testing tokenizer "
        "abababab aaaaaa"
    )

    texts = [
        "",
        "hello",
        "hello world",
        "Hello, world!",
        "abababab",
        "aaaaaa",
        "unknown words",
        "123456789",
        "é😊",
        "你好世界",
        "multiple   spaces",
        "leading and trailing spaces ",
        "\nNew line!",
        "tabs\tbetween\twords",
    ]

    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text, (
            f"Round-trip failed for {text!r}: "
            f"expected {text!r}, got {decoded!r}"
        )

    print("Encode-decode round-trip test passed")

    print("\nAll decoding tests passed!")

if __name__ == "__main__":
    test_pre_tokenize()
    test_merge_pair()
    test_train()
    test_encode()
    test_encode_against_reference()
    test_decode()