A raw read_file tool is surgical but blind: the model must already guess the right path and then wade through the entire file. It doesn’t help discover relevant code when the model doesn’t know where to look. It also tends to bloat prompts with full files or noisy sections.

RAG (chunk + index + retrieve) buys:

Targeted retrieval: breaks files into scoped chunks (functions/blocks) so you pull only the relevant parts.
Discovery: semantic/lexical search over the whole repo so the agent can find code it hasn’t opened yet without guessing file names.
Noise control: injects concise snippets instead of whole files, keeping context lean and focused.
Ranking + evidence: returns scored snippets (path + lines), so you know why you’re injecting them and can log the retrieval trail.
Eval hooks: retrieval metrics (hit rate, overlap with gold answers) let you debug failures: was it bad retrieval or bad generation?
Net: RAG complements read_file. read_file is for precise inspection once you know the target; RAG is for finding candidates and supplying minimal, relevant context so the agent stays grounded and the prompt stays clean.