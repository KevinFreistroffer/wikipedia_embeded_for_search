import wikipedia  # for downloading example Wikipedia articles
import logging
import mwparserfromhell  # for splitting Wikipedia articles into sections
from openai import OpenAI  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import asyncio
from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPEN_API_KEY:
    raise ValueError("Missing OpenAI key")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPEN_API_KEY)

GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 1000
MAX_TOKENS = 1600
CATEGORY_TITLE = "Category:2022 Winter Olympics"
WIKI_SITE = "en.wikipedia.org"
SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
]

if not client:
    raise ValueError("Missing OpenAI key")


def get_all_page_titles_of_category(category_name: str, max_depth: int = 1) -> set[str]:
    """
    Return the page title and linked pages titles of this category.
    """

    titles = set()

    try:
        # Get all pages in the category
        pages = wikipedia.page(category_name, auto_suggest=False)

        # Add the main page
        logger.info(f"Main page title {pages.title}")
        titles.add(pages.title)
        # Get all links from the page
        for i, link in enumerate(pages.links):
            try:
                # Try to get each linked page
                if i % 10 == 0:
                    logger.info(f"Link at index {i}: {link}")
                linked_page = wikipedia.page(link, auto_suggest=False)
                titles.add(linked_page.title)

                # If we want to go deeper and this is a category page
                if max_depth > 0 and "Category:" in link:
                    logger.info(f"Category: is in link")
                    subcategory_name = link.replace("Category:", "")
                    deeper_titles = get_all_page_titles_of_category(
                        subcategory_name, max_depth - 1
                    )
                    titles.update(deeper_titles)
            except wikipedia.exceptions.PageError:
                continue
            except wikipedia.exceptions.DisambiguationError:
                continue

        logger.info(f"Found titles: {titles}")

    except wikipedia.exceptions.PageError:
        logger.error(f"Category {category_name} does not exist")
    except wikipedia.exceptions.DisambiguationError as e:
        # If it's a disambiguation page, get all the options
        for option in e.options:
            try:
                option_page = wikipedia.page(option, auto_suggest=False)
                titles.add(option_page.title)
            except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
            ):
                continue

    return titles


def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,
    parent_titles: list[str],
    sections_to_ignore: set[str],
) -> list[tuple[list[str], str]]:
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        # ^wiki headings are wrapped like "== Heading =="
        return []
    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(
                all_subsections_from_section(subsection, titles, sections_to_ignore)
            )
        return results


def all_subsections_from_title(
    title: str,
    sections_to_ignore: set[str] = set(SECTIONS_TO_IGNORE),
    site_name: str = WIKI_SITE,
) -> list[tuple[list[str], str]]:
    """From a Wikipedia page title, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """
    try:
        # Get the page content using wikipedia package
        page = wikipedia.page(title, auto_suggest=False)
        text = page.content  # This gets the full content of the page

        # Parse the text using mwparserfromhell
        parsed_text = mwparserfromhell.parse(text)
        headings = [str(h) for h in parsed_text.filter_headings()]

        # Get the summary text (content before first heading)
        if headings:
            summary_text = str(parsed_text).split(headings[0])[0]
        else:
            summary_text = str(parsed_text)

        # Start with the summary
        results = [([title], summary_text)]

        # Process all level 2 sections
        for subsection in parsed_text.get_sections(levels=[2]):
            results.extend(
                all_subsections_from_section(subsection, [title], sections_to_ignore)
            )

        return results

    except wikipedia.exceptions.PageError:
        logger.error(f"Page '{title}' not found")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        logger.info(
            f"'{title}' is a disambiguation page. Using first option: {e.options[0]}"
        )
        # Recursively call with the first option
        return all_subsections_from_title(e.options[0], sections_to_ignore, site_name)
    except Exception as e:
        logger.error(f"Error processing page '{title}': {str(e)}")
        return []


def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)


def keep_section(section: tuple[list[str], str]) -> bool:
    titles, text = section
    if len(text) < 16:
        return False
    return True


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimter(string: str, delimiter: str = "\n") -> list[str, str]:
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        # than there's only 1 paragraph or section.
        return [
            string,
            "",
        ]  # trying to split text into 2, so [section1, nextSection]. Since only 1 paragraph/section, return it, with no next section
    elif len(chunks) == 2:  # already split into 2 sections
        return chunks
    else:
        total_tokens = num_tokens(string)  # ex: text has 160 tokens
        halfway = total_tokens  # ideally, splitting it 80/80 is best
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = "\n".join(chunks[0 : i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)  # 80 - 50 = 30, or 80 - 90 = 10
            if diff >= best_diff:
                break
            else:
                best_diff = diff  # otherwise, this current joined text total tokens is the best difference
        left = "\n".join(
            chunks[:i]
        )  # i is available outside the for loop after it finishes looping. So if it breaks early, than say it breaks at 1, than the text grabs up to that, else that and beyond.
        right = "\n".join(chunks[i:])

        return [left, right]


def truncated_string(
    string: str, model: str, max_tokens: int, print_warning: bool = True
) -> str:
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        logger.warning(
            f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens."
        )
    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    print("subsectionzzzzzz: ", type(subsection))
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)

    if num_tokens_in_string <= max_tokens:
        return [string]
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimter(text, delimiter=delimiter)
            if left == "" or right == "":
                continue
            else:
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    return [truncated_string(string, model=model, max_tokens=max_tokens)]


def get_embeddings(text: str, model: str = GPT_MODEL) -> list[float]:
    client = OpenAI(api_key=os.environ.get("OPEN_API_KEY", ""))
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


async def main():
    try:
        # Set the language to English
        wikipedia.set_lang("en")

        # Get all page titles (category page and linked page titles) in the category
        titles = get_all_page_titles_of_category(CATEGORY_TITLE)
        logger.info(f"Found {len(titles)} pages in category {CATEGORY_TITLE}:")

        # Process each page and get its subsections
        all_subsections = []
        for i, title in enumerate(sorted(titles)):
            if i % 10 == 0:
                logger.info(f"\nProcessing item {i}")
            logger.info(f"\nProcessing: {title}")
            subsections = all_subsections_from_title(title)
            all_subsections.extend(subsections)
            logger.info(f"Found {len(subsections)} subsections")

        logger.info(f"Total subsections found: {len(all_subsections)}")

        wikipedia_sections = []
        for title in titles:
            wikipedia_sections.extend(all_subsections_from_title(title))
        logger.info(f"Total wikipedia sections found: {len(wikipedia_sections)}")
        wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]

        original_num_sections = len(wikipedia_sections)
        wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
        logger.info(
            f"Filtered {original_num_sections - len(wikipedia_sections)} sections"
        )

        wikipedia_strings = []
        for section in wikipedia_sections:
            wikipedia_strings.extend(
                split_strings_from_subsection(section, max_tokens=MAX_TOKENS)
            )

        logger.info(
            f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings."
        )

        embeddings = []
        for batch_start in range(0, len(wikipedia_strings), EMBEDDING_BATCH_SIZE):
            logger.info(
                f"batch_start: {batch_start} EMBEDDING_BATCH_SIZE: {EMBEDDING_BATCH_SIZE}"
            )
            batch_end = batch_start + EMBEDDING_BATCH_SIZE  # 1000
            batch = wikipedia_strings[batch_start:batch_end]
            logger.info(f"batch:")
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            logger.info(f"response: {response}")
            for i, be in enumerate(response.data):
                assert i == be.index
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)

        df = pd.DataFrame({"text": wikipedia_strings, "embeddings": embeddings})

        SAVE_PATH = "data/winter_olypics_2022.csv"
        df.to_csv(SAVE_PATH, index=False)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
