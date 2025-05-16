# Embed Wikipedia articles, saving to a CSV
- Follows this guideline except I replaced mwclient with wikipedia. https://github.com/openai/openai-cookbook/blob/45963434ddf20360d1ff2e9583ef019f4d278f67/examples/Embedding_Wikipedia_articles_for_search.ipynb

## What it does?
- Takes a wikipedia category and visits the page.
- Extracts the page title, sub section titles and content.
- Visits each link and gets that pages title, sub section titles and content.
- Splits oversized sections into smaller sections. 
- Embeds text saves the original text and the embeddings in a CSV.


