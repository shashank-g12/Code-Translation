# TransCoT : Prompt Crafting for LLMs to Achieve Accurate and Effective Code Translation

- We introduce CodeTransCoT, a novel Chain-of-Thought based technique designed to efficiently and accurately translate code from one language to another while preserving structural similarity.
- We propose a Quality metric to measure structural similarity and verify if the generated program matches the intended output. This involves performing static analysis of the code and extracting call graph

The approach and code can be found in `TransCoT` folder. `QualityEval` contains the code to evaluate quality measure of the generated programs using `TransCoT`.
