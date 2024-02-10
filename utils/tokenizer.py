import os


class Tokenizer:
    def __init__(self):
        self.tokens_dict = {}
        self.reverse_tokens_dict = {}

    @property
    def vocab_size(self):
        return len(self.tokens_dict)

    def discover_tokens(self, text: str):
        for token in text:
            token = token.lower()
            if token not in self.tokens_dict:
                index = len(self.tokens_dict)
                self.tokens_dict[token] = index
                self.reverse_tokens_dict[index] = token

    def encode(self, text: str) -> list[int]:
        return [self.tokens_dict[token.lower()] for token in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.reverse_tokens_dict[token] for token in tokens])


def build_tokenizer(data_dir: str) -> Tokenizer:
    tokenizer = Tokenizer()

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r") as f:
            text = f.read()
            tokenizer.discover_tokens(text)

    return tokenizer
