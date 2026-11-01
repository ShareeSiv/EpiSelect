# Common Pile domain constants, shared by the tfds builder and the compact (uint16 memmap) builder.
# Kept free of tensorflow / transformers imports so the compact training path does not need them.

# changing any constant below requires bumping VERSION and rebuilding
TOKEN_CAP = 2_000_000_000  # max tokens per domain
VAL_FRACTION = 0.01

# measured (doc count, mean tokens/doc) per *_filtered domain
DOMAIN_STATS = {
    "arxiv_abstracts":                 (   2_504_679,       183),
    "arxiv_papers":                    (     309_328,    18_503),
    "biodiversity_heritage_library":   (  16_456_245,       550),
    "caselaw":                         (   5_502_674,     3_024),
    "cc_common_crawl":                 (  10_827_540,     1_173),
    "data_provenance":                 (   3_508_390,       289),
    "doab":                            (     406_517,     7_217),
    "foodista":                        (      65_700,       324),
    "github":                          (  23_306_158,       494),
    "gutenberg":                       (      57_115,    92_243),
    "library_of_congress":             (     128_324,    64_878),
    "libretexts":                      (      40_049,     2_480),
    "news":                            (     126_673,       947),
    "oercommons":                      (       5_249,       981),
    "pes2o":                           (   6_088_479,     6_338),
    "pre_1929_books":                  (     122_426,    87_305),
    "pressbooks":                      (      54_455,       812),
    "public_domain_review":            (       1_406,       874),
    "pubmed":                          (   4_768_725,     7_446),
    "python_peps":                     (         655,     4_129),
    "regulations":                     (     192_436,    13_502),
    "stackexchange":                   (  27_497_605,       805),
    "stackv2_edu":                     (  57_038_024,     1_019),
    "stackv2_html":                    (   1_674_654,       708),
    "ubuntu_irc":                      (     216_197,     8_524),
    "uk_hansard":                      (      47_909,    10_716),
    "usgpo":                           (   2_339_475,     3_455),
    "uspto":                           (  14_410_217,     9_484),
    "wikimedia":                       (  12_899_840,     1_018),
    "wikiteam":                        (  10_230_749,       362),
    "youtube":                         (     986_402,     4_143),
}


def split_rows(domain, token_cap=TOKEN_CAP):
    # (val, train) document counts: the first val_rows docs of the HF stream are held out
    n_docs, mean_tokens = DOMAIN_STATS[domain]
    budget_rows = min(n_docs, round(token_cap / mean_tokens))
    val_rows = max(1, round(VAL_FRACTION * budget_rows))
    train_rows = budget_rows - val_rows
    assert train_rows > 0, f"{domain}: no train rows left after the val holdout"
    return val_rows, train_rows


TASKS = [
    "arxiv_abstracts",
    "arxiv_papers",
    "biodiversity_heritage_library",
    "caselaw",
    "cc_common_crawl",
    "data_provenance",
    "doab",
    "foodista",
    "github",
    "gutenberg",
    "library_of_congress",
    "libretexts",
    "news",
    "oercommons",
    "pes2o",
    "pre_1929_books",
    "pressbooks",
    "public_domain_review",
    "pubmed",
    "python_peps",
    "regulations",
    "stackexchange",
    "stackv2_edu",
    "stackv2_html",
    "ubuntu_irc",
    "uk_hansard",
    "usgpo",
    "uspto",
    "wikimedia",
    "wikiteam",
    "youtube",
]

HF_NAMES = {
    "arxiv_abstracts":               "common-pile/arxiv_abstracts_filtered",
    "arxiv_papers":                  "common-pile/arxiv_papers_filtered",
    "biodiversity_heritage_library": "common-pile/biodiversity_heritage_library_filtered",
    "caselaw":                       "common-pile/caselaw_access_project_filtered",
    "cc_common_crawl":               "common-pile/cccc_filtered",
    "data_provenance":               "common-pile/data_provenance_initiative_filtered",
    "doab":                          "common-pile/doab_filtered",
    "foodista":                      "common-pile/foodista_filtered",
    "github":                        "common-pile/github_archive_filtered",
    "gutenberg":                     "common-pile/project_gutenberg_filtered",
    "library_of_congress":           "common-pile/library_of_congress_filtered",
    "libretexts":                    "common-pile/libretexts_filtered",
    "news":                          "common-pile/news_filtered",
    "oercommons":                    "common-pile/oercommons_filtered",
    "pes2o":                         "common-pile/peS2o_filtered",
    "pre_1929_books":                "common-pile/pre_1929_books_filtered",
    "pressbooks":                    "common-pile/pressbooks_filtered",
    "public_domain_review":          "common-pile/public_domain_review_filtered",
    "pubmed":                        "common-pile/pubmed_filtered",
    "python_peps":                   "common-pile/python_enhancement_proposals_filtered",
    "regulations":                   "common-pile/regulations_filtered",
    "stackexchange":                 "common-pile/stackexchange_filtered",
    "stackv2_edu":                   "common-pile/stackv2_edu_filtered",
    "stackv2_html":                  "common-pile/stackv2_html_filtered",
    "ubuntu_irc":                    "common-pile/ubuntu_irc_filtered",
    "uk_hansard":                    "common-pile/uk_hansard_filtered",
    "usgpo":                         "common-pile/usgpo_filtered",
    "uspto":                         "common-pile/uspto_filtered",
    "wikimedia":                     "common-pile/wikimedia_filtered",
    "wikiteam":                      "common-pile/wikiteam_filtered",
    "youtube":                       "common-pile/youtube_filtered",
}


TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
_tokenizer = None


def get_tokenizer():
    # loaded on first use: importing this module must not hit the network
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer
