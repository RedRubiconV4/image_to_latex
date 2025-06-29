import re
import pandas as pd
from collections import Counter

class LaTeXTokeniser:
    def __init__(self):
        # Define patterns for LaTeX commands, symbols, numbers, letters, and mathematical functions
        self.command_pattern = r'\\[a-zA-Z]+'
        self.number_pattern = r'\d+\.\d+|\d+'  # Matches integers and floats
        self.letter_pattern = r'[a-zA-Z]+'  # Matches letters (both lowercase and uppercase)
        self.symbol_pattern = r'[\+\-=\*/\^\(\)\{\}\[\]\_,\.;]'  # Symbols
        self.function_pattern = r'\\[a-zA-Z]+(?=\s?\()'  # Functions like \sin, \cos, \exp (detects leading \)
        self.subscript_pattern = r'_'
        self.superscript_pattern = r'\^'
        self.brace_pattern = r'\{|\}'

        # Combine all token patterns into a list
        self.token_patterns = [
            self.command_pattern,
            self.number_pattern,
            self.letter_pattern,
            self.function_pattern,
            self.symbol_pattern,
            self.subscript_pattern,
            self.superscript_pattern,
            self.brace_pattern
        ]

    def tokenise(self, latex_string):
        tokens = []
        combined_pattern = '|'.join(self.token_patterns)
        matches = re.finditer(combined_pattern, latex_string)

        for match in matches:
            tokens.append(match.group(0))

        return tokens