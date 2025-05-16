# Embed Wikipedia articles, saving to a CSV
- Follows this guideline, replacing mwclient with wikipedia. https://github.com/openai/openai-cookbook/blob/45963434ddf20360d1ff2e9583ef019f4d278f67/examples/Embedding_Wikipedia_articles_for_search.ipynb
- Takes a wikipedia category, gets the page title, gets all links on the page and gets those pages titles.
- Gets each titles content and sub section titles content. Splits large section into smaller sections. 
- Creates embeddings and saves the original text and the embeddings in a CSV.
- Should save to a vector database with larger data.
