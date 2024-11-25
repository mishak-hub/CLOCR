The following 7 quoted texts are prompt templates for an LLM. They include curly braces as a formatting space for text that will be inserted later. Please first summarize all 7 prompts. Then, be an expert prompt engineer and construct one unified prompt using the best features, details, and descriptions from the 7 original prompts. You are instructed to be clear and descriptive.

"""### Instruction:
Fix the OCR errors in the provided text.

### Input:
{}

### Response:
{}
""",

"""### Instruction:
The following sentence has grammatical errors, strange words, and spelling mistakes introduced by OCR scanning. Please rewrite it correctly and clearly, without adding new or extraneous tokens.

### Input:
{}

### Response:
{}
""",

"""### Instruction:
The provided text contains mistakes due to OCR processing. Please correct it for readability and accuracy by fixing any errors introduced by OCR, ensuring the result is grammatically correct and clear. Keep the original meaning intact.

### Input:
{}

### Response:
{}
""",

"""### Instruction:
Edit the following sentence to correct OCR mistakes without altering the total number of characters in the original scanned document. This number is unknown to you, but can be assumed to be the number of characters if everything is spelled correctly and extraneous symbols or corrupted text had its noise rectified.

### Input:
{}

### Response:
{}
""",

"""### Instruction:
The following input contains mistakes introduced solely by OCR. Please fix any spelling, grammar, or formatting issues in the provided text without adding or removing characters from the original document scanned by OCR.

### Input:
{}

### Response:
{}
""",

"""### Instruction:
Revise the following text to correct OCR errors, but maintain the exact character count and avoid introducing any new characters.

### Input:
{}

### Response:
{}
""",

"""### Instruction:
Correct only the errors in the provided text. Avoid adding or removing any characters, and keep the character count consistent with what the original document would look like without error noise.

### Input:
{}

### Response:
{}
"""


------------------------------------
Summary and new Prompt 8 from GPT-4o
------------------------------------

### Summary of the 7 Prompts:

1. **Prompt 1**: General instruction to fix OCR errors in the provided text, leaving the response open-ended. Focus on "fixing errors" without specific constraints.
2. **Prompt 2**: Requests rewriting the text to correct grammatical errors, strange words, and spelling mistakes from OCR. Emphasizes clarity and correctness while avoiding extraneous tokens.
3. **Prompt 3**: Corrects OCR mistakes to ensure readability and grammatical accuracy while preserving the original meaning.
4. **Prompt 4**: Corrects OCR mistakes while retaining the same total number of characters as the original text, with an implied but undefined target for accurate representation.
5. **Prompt 5**: Fixes OCR issues while maintaining the exact number of characters, focusing on spelling, grammar, and formatting errors.
6. **Prompt 6**: Requires corrections to OCR errors without introducing or removing any characters, keeping the original character count consistent.
7. **Prompt 7**: Emphasizes error correction without changing the character count to match the intended appearance of the original document.

---

### Unified Prompt:

```plaintext
"""### Instruction:
The following text contains errors introduced by OCR scanning, such as misspellings, grammatical mistakes, and formatting issues. Your task is to correct the text to ensure it is grammatically accurate, readable, and true to its original intended meaning.  

**Constraints:**
1. Maintain the exact character count of the corrected text to match what the original, error-free document would contain.
2. Do not introduce any new characters, words, or symbols that were not part of the original input.
3. Correct only the OCR errors without altering the structure or content unnecessarily.  

### Input:
{}

### Response:
{}
"""
``` 

### Features Incorporated:
- **From Prompt 1**: General task of fixing OCR errors.
- **From Prompt 2**: Clarity and correctness, avoiding extraneous additions.
- **From Prompt 3**: Focus on preserving the original meaning.
- **From Prompts 4-7**: Character count constraints and emphasis on precision, ensuring no characters are added or removed.


