import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory) -> dict:
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus: dict, page: list, damping_factor: float) -> dict:
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    total_pages = len(corpus)

    # Probability of following a link from the current page
    if len(corpus[page]) > 0:
        link_probability = damping_factor / len(corpus[page])

    # Probability of choosing any page at random
    random_probability = (1 - damping_factor) / total_pages

    # Initialize the probability distribution
    distribution = {}

    # If the current page has no outgoing links, treat it as if it links to all pages
    if len(corpus[page]) == 0:
        return {p: 1 / total_pages for p in corpus}

    # else
    for p in corpus:
        if p in corpus[page]:
            distribution[p] = random_probability + link_probability
        else:
            distribution[p] = random_probability

    return distribution


def sample_pagerank(corpus: dict, damping_factor: float, n: int) -> dict:
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    counts = {page: 0 for page in corpus}

    # Start with a random page
    current_page = random.choice(list(corpus.keys()))

    # Perform `n` samples
    for _ in range(n):
        # Increment the count for the current page
        counts[current_page] += 1

        # Get the transition model for the current page
        distribution = transition_model(corpus, current_page, damping_factor)

        # Choose the next page based on the distribution
        keys, values = list(distribution.keys()), list(distribution.values())
        current_page = random.choices(keys, values)[0]

    # Normalize counts to get the PageRank values
    total_samples = sum(counts.values())

    ranks = {}
    for page, count in counts.items():
        ranks[page] = count / total_samples

    return ranks


def iterate_pagerank(corpus: dict, damping_factor: float) -> dict:
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    threshold: float = 0.001
    ranks: dict = {page: 1 / len(corpus) for page in corpus}

    while True:
        new_ranks: dict = {}
        for page in corpus:
            rank_sum: int = 0
            for link in corpus:
                if page in corpus[link] and len(corpus[link]) > 0:
                    rank_sum += ranks[link] / len(corpus[link])

                if len(corpus[link]) / len(corpus) == 0:
                    rank_sum += ranks[link] / len(corpus)

            # Compute new rank by applying the PageRank formula
            new_ranks[page] = (1 - damping_factor) / len(
                corpus
            ) + damping_factor * rank_sum

        if all(abs(new_ranks[page] - ranks[page]) < threshold for page in ranks):
            break

        ranks = new_ranks

    return ranks


if __name__ == "__main__":
    main()
