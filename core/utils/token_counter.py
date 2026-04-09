class ClaudeTokenCounter:

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text for Claude.
        This is basic implementation and is not accurate
        """

        return len(text) // 3 if text else 0
