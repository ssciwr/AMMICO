"""Helpers for the two question-accuracy notebooks in this folder:

* question_accuracy_single_model.ipynb - compare the questions for one model
* question_accuracy_models.ipynb       - compare the questions for several models

The notebooks only need the model result file(s) and the image folder. Everything else
(which dataset it is, the questions, the ground truth, the expected answers) is detected
from the result files and taken from the dataset descriptions in DATASETS below.

1. datasets      - DATASETS: one entry per course dataset (questions, ground truth, expected answers)
2. loading       - result files in three formats (detected automatically), dataset detection
3. normalising   - free-text answers -> categories (yes / no / dk / ...), with the rule that was used
4. scoring       - expected answer from the ground truth, accuracy (DK = wrong), Wilson CI,
                   exact McNemar test between questions
5. reporting     - one show_* function per notebook section: tables, plots and a result sentence
"""

import ast
import glob
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 200)

ROOT = (
    Path(__file__).resolve().parent.parent
)  # repository root (or the workshop folder)
# Folders searched for the ground-truth and question files, in this order:
#   <folder of IMAGE_DIR>/../ground_truth   (workshop folder: dataset_<name>/ next to ground_truth/)
#   ROOT/ground_truth                        (notebooks in a folder next to ground_truth/)
#   ROOT/<dataset>_output                    (this repository)
# Set GROUND_TRUTH_DIR to use another folder first.
GROUND_TRUTH_DIR = None

# ---------------------------------------------------------------------------
# 1. Datasets
#
# Each dataset describes:
#   folder           folder of the dataset in this repository (see GROUND_TRUTH_DIR for the search order)
#   labels           ground-truth csv file name (column `filename` + label columns)
#   label_column     column the expected answers are derived from
#   questions        file name of the questions (.txt with a python list, or .csv with a `prompt` column)
#   types            one question type per question; selects the normalisation rules (see NORMALISERS)
#   styles           short description of each wording, used in the plot labels
#   expected         {question type: {label value: expected answer}}; label values without an
#                    entry (e.g. 'unclear') are not scored
# optional:
#   label_columns    ground-truth column per question (default: label_column for all)
#   path_labels      {new column: text} -> "yes" if the image path contains the text, else "no"
#                    (only if the ground truth has no such column)
#   group_column     column that defines the image groups (default: the label)
#   group_rule       function(ground-truth table) -> group per image, instead of group_column
#   group_order      order of the groups in the examples
#   subfolder_column column with the image folder (default: folder of the image path)
#   summary_questions questions compared in the summary (default: all)
#
# The dataset of a result file is the one whose ground truth contains most of its images.
# ---------------------------------------------------------------------------
DATASETS = {
    "emotions": {
        "folder": "emotions_output",
        "labels": "emotions_labels.csv",
        "label_column": "emotion",
        "questions": "emotions_questions16NOV26.txt",
        "types": [
            "polarity",
            "polarity",
            "negative",
            "negative",
            "negative",
            "positive",
            "positive",
            "positive",
        ],
        "styles": [
            "open",
            "constrained",
            "open",
            "constrained",
            "constrained+cues",
            "open",
            "constrained",
            "constrained+cues",
        ],
        "expected": {
            "polarity": {
                "positive": "positive",
                "negative": "negative",
                "neutral": "none",
            },
            "negative": {"positive": "no", "negative": "yes", "neutral": "no"},
            "positive": {"positive": "yes", "negative": "no", "neutral": "no"},
        },
        "group_column": "emotion",
        "group_order": ["negative", "neutral", "positive"],
        "subfolder_column": "subfolder",
    },
    "politicians": {
        "folder": "politicians_output",
        "labels": "politicians_labels.csv",
        "label_column": "politician",
        "questions": "politicians_questions17NOV26.txt",
        "types": ["politician", "politician", "politician", "named"],
        "styles": ["open", "constrained", "constrained+justify", "constrained"],
        # Q4 asks for Trump / Zelenskyy / Putin: its expected answer comes from the column
        # named_politician of the ground truth (none / trump / zelenskyy / putin)
        "expected": {
            "politician": {"yes": "yes", "no": "no"},
            "named": {"none": "no", "trump": "yes", "zelenskyy": "yes", "putin": "yes"},
        },
        "label_columns": ["politician", "politician", "politician", "named_politician"],
        "group_rule": lambda lab: np.select(
            [
                lab.named_politician.isin(["trump", "zelenskyy", "putin"]),
                lab.politician == "yes",
            ],
            ["Trump/Zelenskyy/Putin", "other politician"],
            "non-politician",
        ),
        "group_order": ["Trump/Zelenskyy/Putin", "other politician", "non-politician"],
        "summary_questions": [
            "Q1",
            "Q2",
            "Q3",
        ],  # Q4 asks a different, imbalanced question
    },
    "symbols": {
        "folder": "symbols_output",
        "labels": "symbol_labels.csv",
        "label_column": "human_nat_eu_symbol",
        "questions": "symbols_questions16NOV26.txt",
        "types": ["symbol"] * 4,
        "styles": ["open", "examples", "examples+partial", "partial+blended"],
        "expected": {"symbol": {"1": "yes", "0": "no"}},
        "group_column": "human_category",
        "group_order": ["eu_flag", "other_national_or_eu_symbol", "no_symbol"],
        "subfolder_column": "original_folder",
    },
}

# Active question set (set by setup_questions)
QIDS = []
QMETA = pd.DataFrame(columns=["target", "style", "question", "label_col"])
EXPECTED = {}


def _find_file(name, folder, image_dir=None):
    """First existing `name` in the ground-truth folders (see GROUND_TRUTH_DIR); the last candidate if none exists."""
    dirs = [GROUND_TRUTH_DIR] if GROUND_TRUTH_DIR else []
    if image_dir:
        dirs.append(Path(image_dir).expanduser().resolve().parent / "ground_truth")
    dirs += [ROOT / "ground_truth", ROOT / folder]
    candidates = [Path(d).expanduser() / name for d in dirs]
    return next((c for c in candidates if c.exists()), candidates[-1])


def resolve_datasets(image_dir=None):
    """DATASETS with the full paths of the ground-truth and question files."""
    return {
        name: dict(
            cfg,
            labels=_find_file(cfg["labels"], cfg["folder"], image_dir),
            questions=_find_file(cfg["questions"], cfg["folder"], image_dir),
        )
        for name, cfg in DATASETS.items()
    }


def read_questions(path, n):
    """Questions from a .txt file (a python list, or quoted strings separated by commas without brackets),
    or from the `prompt` column of a .csv file.
    Falls back to placeholders if the file is not there (the texts are only displayed)."""
    path = Path(path)
    if not path.exists():
        return [f"(question text not found: {path.name} is missing)"] * n
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)["prompt"].astype(str).tolist()
    txt = path.read_text()
    txt = txt[txt.index("[") : txt.rindex("]") + 1] if "[" in txt else f"[{txt}]"
    return list(ast.literal_eval(txt))


def setup_questions(questions, types, expected, styles=None, label_columns=None):
    """Define the active question set (Q1, Q2, ... in the order of `questions`)."""
    global QIDS, QMETA, EXPECTED
    n = len(questions)
    styles = list(styles) if styles else [""] * n
    label_columns = list(label_columns) if label_columns else ["label"] * n
    assert len(types) == n == len(styles) == len(label_columns), (
        f"{n} questions, but {len(types)} types / {len(styles)} styles / {len(label_columns)} label columns"
    )
    unknown = set(types) - set(NORMALISERS)
    assert not unknown, (
        f"unknown question type(s) {unknown}; available: {list(NORMALISERS)}"
    )
    QIDS = [f"Q{i}" for i in range(1, n + 1)]
    QMETA = pd.DataFrame(
        {
            "qid": QIDS,
            "target": list(types),
            "style": styles,
            "question": questions,
            "label_col": label_columns,
        }
    ).set_index("qid")
    EXPECTED = {t: {str(k): v for k, v in m.items()} for t, m in expected.items()}
    return QMETA


def qlabel(q):
    style = QMETA.loc[q, "style"]
    return f"{q}\n{QMETA.loc[q, 'target']}" + (f"/{style}" if style else "")


# ---------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------
def _short_model(name):
    return str(name).split("/")[-1]


def _expand(paths):
    """Paths or glob patterns -> sorted list of existing files."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files = []
    for p in paths:
        hits = sorted(glob.glob(str(Path(p).expanduser())))
        assert hits, f"result file not found: {p}"
        files += [Path(h) for h in hits]
    return files


def _detect_format(raw):
    """'vqa_list' (column vqa), 'coded' (<key>__q<i>_code columns) or 'wide' (one list column per model)."""
    if "vqa" in raw.columns:
        return "vqa_list", None
    keys = list(
        dict.fromkeys(
            m.group(1) for c in raw.columns if (m := re.match(r"(.+)__q\d+_code$", c))
        )
    )
    if keys:
        return "coded", keys
    cols = [
        c
        for c in raw.columns
        if c != "filename"
        and not c.startswith("Unnamed")
        and raw[c].dropna().astype(str).str.lstrip().str.startswith("[").mean() > 0.5
    ]
    assert cols, (
        f"cannot read the answers: no 'vqa' column, no coded columns and no list columns in {list(raw.columns)}"
    )
    return "wide", cols


def _read_answers(raw, path):
    """One table per model: filename, source_path, model, Q1..Qn (raw answers, None = missing)."""
    base = pd.DataFrame(
        {
            "filename": raw["filename"].map(os.path.basename),
            "source_path": raw["filename"],
        }
    )
    fmt, cols = _detect_format(raw)
    frames = []
    if fmt == "vqa_list":
        models = (
            raw["model"]
            if "model" in raw.columns
            else pd.Series(Path(path).stem, index=raw.index)
        )
        answers = raw["vqa"].map(parse_vqa_list)
        for m in models.unique():
            out = base[models.values == m].copy()
            out["model"] = m
            for i, q in enumerate(QIDS):
                out[q] = answers[models.values == m].map(lambda a, i=i: a[i])
            frames.append(out)
    elif fmt == "wide":
        for m in cols:
            out = base.copy()
            out["model"] = m
            answers = raw[m].map(parse_vqa_list)
            for i, q in enumerate(QIDS):
                out[q] = answers.map(lambda a, i=i: a[i])
            frames.append(out)
    else:  # coded (codes 1 = yes, 0 = no, 9 = dk, 8 = unclear)
        for key in cols:
            out = base.copy()
            out["model"] = key
            for i, q in enumerate(QIDS, start=1):
                out[q] = raw[f"{key}__q{i}_raw"].where(
                    raw[f"{key}__q{i}_raw"].notna(), None
                )
                out[q + "_code"] = raw[f"{key}__q{i}_code"]
            frames.append(out)
    return frames


def detect_dataset(filenames, datasets=None):
    """The dataset whose ground truth contains the largest share of the given image file names."""
    datasets = datasets or resolve_datasets()
    filenames = set(filenames)
    share = {
        name: len(
            filenames
            & set(
                pd.read_csv(cfg["labels"], encoding="utf-8-sig")["filename"]
                .astype(str)
                .str.strip()
                .map(os.path.basename)
            )
        )
        / len(filenames)
        for name, cfg in datasets.items()
        if Path(cfg["labels"]).exists()
    }
    assert share, (
        "no ground-truth file found; searched for "
        + ", ".join(str(cfg["labels"]) for cfg in datasets.values())
        + " (set U.GROUND_TRUTH_DIR to the folder with the label csv files)"
    )
    best = max(share, key=share.get)
    assert share[best] >= 0.5, (
        f"the result images do not match any known dataset (share found in the ground "
        f"truth: {share}); add your dataset to DATASETS in vqa_eval_utils.py"
    )
    return best


def load_ground_truth(cfg, results, images):
    """Ground-truth table with the columns filename, label, group, subfolder (+ all original columns)."""
    lab = pd.read_csv(cfg["labels"], encoding="utf-8-sig")
    lab["filename"] = lab["filename"].astype(str).str.strip().map(os.path.basename)
    assert not lab["filename"].duplicated().any(), (
        "duplicate file names in the ground truth"
    )
    lab["label"] = lab[cfg["label_column"]].astype("string").str.strip()
    # Folder-based labels (path_labels) are read from the image path as written in the result file and
    # from the path in IMAGE_DIR (flat folders are fine if the other one has the folders).
    # A column of the same name in the ground truth wins (and is needed with flat folders).
    res_path = (
        lab["filename"]
        .map(
            results.drop_duplicates("filename")
            .set_index("filename")["source_path"]
            .astype(str)
        )
        .fillna("")
    )
    img_path = lab["filename"].map(
        lambda f: images[f].as_posix() if f in images else ""
    )
    p = res_path + "|" + img_path

    def path_label(col, text):
        hit = p.str.contains(text, regex=False)
        if hit.any():
            return np.where(hit, "yes", "no")
        print(
            f"WARNING: the ground truth has no column {col!r} and no image path contains {text!r}, "
            f"so the questions that use {col!r} are not scored. "
            f"With flat folders, add the column to {cfg['labels'].name}."
        )
        return np.nan

    sub = cfg.get("subfolder_column")
    if sub:
        lab["subfolder"] = (
            lab[sub].fillna("(none)").astype(str).map(lambda x: Path(x).name)
        )
    else:  # folder of the image: from whichever path has more distinct folders (i.e. is not flat)
        folders = [
            x.map(lambda v: Path(v).parent.name if v else "(none)")
            for x in (res_path, img_path)
        ]
        lab["subfolder"] = max(folders, key=lambda f: f.nunique())
    for col, text in cfg.get("path_labels", {}).items():
        if col not in lab.columns:
            lab[col] = path_label(col, text)
    if cfg.get("group_rule"):
        lab["group"] = cfg["group_rule"](lab)
    else:
        col = cfg.get("group_column")
        lab["group"] = lab[col].fillna("(none)").astype(str) if col else lab["label"]
    return lab


def load(results, image_dir, output_name):
    """Load the result file(s), detect the dataset, normalise the answers and score them.

    results     : path (or list of paths / glob patterns) of the csv file(s) with the model answers
    image_dir   : folder with the images (searched recursively; only needed for the image examples)
    output_name : name of the output folder (comparison_results/<output_name> next to the first result file)
    Returns a namespace with the data tables, the dataset description and the output folder.
    """
    files = _expand(results)
    raws = {f: pd.read_csv(f, encoding="utf-8-sig") for f in files}
    datasets = resolve_datasets(image_dir)
    dataset = detect_dataset(
        set().union(*(r["filename"].map(os.path.basename) for r in raws.values())),
        datasets,
    )
    cfg = datasets[dataset]
    setup_questions(
        read_questions(cfg["questions"], len(cfg["types"])),
        cfg["types"],
        cfg["expected"],
        cfg.get("styles"),
        cfg.get("label_columns"),
    )

    res = pd.concat(
        [fr for f, raw in raws.items() for fr in _read_answers(raw, f)],
        ignore_index=True,
    )
    res["model"] = res["model"].map(_short_model)
    unanswered = [q for q in QIDS if res[q].isna().all()]
    assert not unanswered, (
        f"the {dataset} dataset has {len(QIDS)} questions, but no model answered {', '.join(unanswered)}: "
        f"the result files seem to come from a different question set "
        f"({len(QIDS) - len(unanswered)} answers per image instead of {len(QIDS)}). "
        f"Run the {len(QIDS)} questions of {cfg['questions'].name}, or change the dataset description in DATASETS."
    )
    dup = res.duplicated(["model", "filename"])
    assert not dup.any(), (
        f"duplicate images for one model: {res[dup][['model', 'filename']].head().values.tolist()}"
    )

    images = image_index(image_dir)
    truth = load_ground_truth(cfg, res, images)
    data = normalise(
        res.merge(truth, on="filename", how="inner", validate="many_to_one")
    )
    models = list(dict.fromkeys(data.model))

    # with several models: score all models on the same images (those every model answered completely)
    complete = data.assign(ok=data[QIDS].notna().all(axis=1)).groupby(
        "filename"
    ).ok.all() & (data.groupby("filename").model.nunique() == len(models))
    common = set(complete[complete].index) if len(models) > 1 else set(data.filename)
    long = to_long(data)
    long = long[long.filename.isin(common)].copy()

    out = files[0].parent / "comparison_results" / output_name
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "answers_long.csv", index=False)

    n_res = res.filename.nunique()
    print(f"dataset: {dataset}  |  models: {', '.join(models)}")
    print(
        f"images: {n_res} in the results, {data.filename.nunique()} with ground truth, "
        f"{len(set(images) & set(data.filename))} found in {image_dir}"
    )
    if len(models) > 1:
        print(
            f"images answered completely by all models (used for the comparison): {len(common)}"
        )
    print(f"output folder: {out}")
    return SimpleNamespace(
        dataset=dataset,
        cfg=cfg,
        models=models,
        data=data,
        long=long,
        truth=truth,
        images=images,
        out=out,
        n_dropped=data.filename.nunique() - len(common),
    )


def parse_vqa_list(s, n=None):
    """Return a list of length n (default: number of questions); entries are None when the answer is missing/empty."""
    n = n or len(QIDS)
    try:
        lst = ast.literal_eval(s) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        lst = []
    lst = [a if isinstance(a, str) and a.strip() else None for a in lst]
    return (lst + [None] * n)[:n]


# ---------------------------------------------------------------------------
# 3. Normalisation
#
# Every normaliser returns (category, rule) so that the mapping can be audited.
# Rules, in order of precedence:
#   missing   - no answer
#   exact     - the whole answer is a category ("Yes.", "Negative", "DK")
#   first     - the first word is a category ("No, the picture ...")
#   answer_is - "... the answer is: yes"
#   kw_*      - keyword rules for free-text answers (see below)
#   unparsed  - nothing matched
# ---------------------------------------------------------------------------
POLARITY_CATS = ["positive", "negative", "both", "none", "dk"]
YESNO_CATS = ["yes", "no", "dk"]

# hedging / can't tell
DK_PAT = re.compile(
    r"^(dk|i don'?t know|don'?t know|do not know|unsure|not sure)$"
    r"|difficult to (definitively |accurately |clearly )?(determine|assess|say|tell|judge|interpret)"
    r"|(can'?t|cannot|unable to) (be )?(definitively |accurately )?(determine|assess|tell|judge)"
    r"|cannot be (definitively |accurately )?(determined|assessed)"
    r"|(not|n'?t) (provide )?(enough|sufficient) (context|information)"
    r"|too (blurry|small|dark)"
    r"|unclear (if|whether)"
    r"|(is|remains) subjective|depends on (the )?(individual|context|viewer)"
)
# "without (more) context" is hedging, not a negation of the emotion
CONTEXT_HEDGE = re.compile(
    r"without (any |more |additional |further |the )?(context|information|visual cues)[^.;]*"
)

# words that signal emotional polarity in free text
NEG_WORDS = (
    r"negativ\w*|somber|sombre|sad\w*|tense|tension|conflict\w*|fear\w*|anger|angry|distress\w*"
    r"|grief|danger\w*|chao\w*|destruct\w*|devastat\w*|violen\w*|frustrat\w*|concern\w*|pain\w*"
    r"|loss|tragic|alarm\w*|disappoint\w*|melanchol\w*|solemn\w*|unrest|hardship|anxi\w*|worr\w*"
    r"|disapprov\w*|critical|shock\w*|upset"
)
POS_WORDS = (
    r"positiv\w*|happ\w*|joy\w*|celebrat\w*|cheerful|smil\w*|laugh\w*|warm|friendly|triumph\w*"
    r"|delight\w*|optimis\w*|enjoy\w*|pleasant|uplifting|cheer\w*|excite\w*|excitement|contentment"
    r"|gratitude|pride|proud|love|affection\w*"
)
NEG_RE = re.compile(rf"\b({NEG_WORDS})\b")
POS_RE = re.compile(rf"\b({POS_WORDS})\b")
# negator followed (in the same clause, <= 60 chars) by an emotion word
NEGATOR = r"\b(not|n't|no|never|without|lack\w*|rather than|nor|neither|absence of|free of|instead of)\b"

# "no emotion at all" (deliberately NOT "does not show any positive emotion" - that says nothing about negative)
_ADJ = r"( explicit| clear| obvious| particular| strong| specific| discernible| overt| apparent| visible)?"
NONE_RE = re.compile(
    r"(does not|doesn'?t|do not|did not|is not)( explicitly| clearly| really| necessarily| inherently)?"
    rf" (contain|show|convey|display|depict|have|evoke|present|express)( any)?{_ADJ}( positive or negative| positive and negative)? emotion"
    rf"|no{_ADJ}( positive or negative)? emotion|there is no{_ADJ}( positive or negative)? emotion"
    r"|without any (clear |explicit |obvious )?emotional content"
    r"|neither positive nor negative|neither overtly positive nor negative|emotionally neutral"
)
NEUTRAL_RE = re.compile(r"\bneutral\b")
HOWEVER_RE = re.compile(r"\b(however|but)\b(.*)")
BOTH_RE = re.compile(
    r"both positive and negative|mix(ed|ture)? of (emotions|positive and negative)|mixed emotion"
)
# explicit conclusion: "Therefore, the (correct) answer is: negative", "Answer: yes", 'the answer would be "none"'
_ANSWER_RE = re.compile(
    r"\banswer(?: is| would be)?(?: that)?\s*:?\s*[\"'*]*\s*(?:the picture shows\s*[\"'*]*)?"
    r"(yes|no|dk|positive|negative|both|none)\b"
)


def _explicit_answer(c):
    """Category of the LAST explicit 'answer is ...' in the text, or None."""
    hits = _ANSWER_RE.findall(c)
    return hits[-1] if hits else None


def _isna(s):
    return s is None or (isinstance(s, float) and math.isnan(s))


def _clean(s):
    s = s.lower().replace("*", "").replace("’", "'").strip()
    return re.sub(r"^[\s\"'`]+|[\s\"'`.!:,]+$", "", s)


def _first_word(c):
    return re.split(r"[\s.,;:!]+", c, maxsplit=1)[0]


def _negated(c, word_re_src):
    return re.search(rf"{NEGATOR}[^.;]{{0,60}}?\b({word_re_src})\b", c) is not None


def norm_polarity(s):
    """Q1/Q2 -> positive | negative | both | none | dk | unparsed | missing."""
    if _isna(s):
        return "missing", "missing"
    c = _clean(s)
    alias = {
        "neutral": "none",
        "mixed": "both",
        "no": "none",
    }  # "No" to "any positive or negative?" = none
    if c in POLARITY_CATS or c in alias:
        return alias.get(c, c), "exact"
    w = _first_word(c)
    if w in POLARITY_CATS or w in ("neutral", "mixed"):
        return alias.get(w, w), "first"
    a = _explicit_answer(c)
    if a and a != "yes":
        return {"no": "none"}.get(a, a), "answer_is"
    if BOTH_RE.search(c):
        return "both", "kw_both"
    m = NONE_RE.search(c)
    if m:
        # "... shows no emotion. However, the text suggests a negative topic" -> follow the 'however' clause
        h = HOWEVER_RE.search(c[m.end() :])
        pol = (
            _first_polarity_keyword(re.split(r"[.;!]", h.group(2), maxsplit=1)[0])
            if h
            else None
        )
        return (pol, "kw_none_however") if pol else ("none", "kw_none")
    if DK_PAT.search(c):
        return "dk", "kw_dk"
    first_sentence = re.split(r"(?<=[.;!])\s", c, maxsplit=1)[0]
    if NEUTRAL_RE.search(first_sentence) and not _first_polarity_keyword(
        first_sentence
    ):
        return "none", "kw_neutral"
    pol = _first_polarity_keyword(c)
    if pol:
        return pol, "kw_polarity"
    if NEUTRAL_RE.search(c):
        return "none", "kw_neutral"
    return "unparsed", "unparsed"


# a negator followed by an emotion word, including chains such as "no clear signs of positive or negative emotions"
_KW = rf"({NEG_WORDS}|{POS_WORDS})"
NEGATED_CHAIN = re.compile(
    rf"{NEGATOR}[^.;]{{0,60}}?\b{_KW}\b(\s*(,|or|and|nor|/)\s*(\w+\s+)?{_KW}\b)*"
)


def _first_polarity_keyword(c):
    """First non-negated emotional keyword wins ("no clear indication of positive ...; rather tension" -> negative)."""
    c2 = NEGATED_CHAIN.sub(" ", c)
    m_neg, m_pos = NEG_RE.search(c2), POS_RE.search(c2)
    if m_neg and (not m_pos or m_neg.start() < m_pos.start()):
        return "negative"
    if m_pos:
        return "positive"
    return None


def norm_yesno(s, target):
    """Q3-Q8 -> yes | no | dk | unparsed | missing.

    target: 'negative' or 'positive' (what the question asks about). Free-text rules:
      1. kw_negated : the asked-for emotion is negated   ("does not show any negative ...") -> no
      2. kw_dk      : hedging                            ("difficult to determine ...")      -> dk
      3. kw_affirm  : the asked-for emotion is mentioned ("... indicating a positive tone")  -> yes
      4. kw_opposite: only the opposite emotion / neutrality is mentioned                    -> no
    """
    if _isna(s):
        return "missing", "missing"
    c = _clean(s)
    if c in YESNO_CATS:
        return c, "exact"
    w = _first_word(c)
    if w in YESNO_CATS:
        return w, "first"
    a = _explicit_answer(c)
    if a in YESNO_CATS:
        return a, "answer_is"
    if c in ("i don't know", "don't know"):
        return "dk", "exact"
    own, other = (
        (NEG_WORDS, POS_WORDS) if target == "negative" else (POS_WORDS, NEG_WORDS)
    )
    c_noctx = CONTEXT_HEDGE.sub(" ", c)
    if _negated(c_noctx, own):
        return "no", "kw_negated"
    if DK_PAT.search(c):
        return "dk", "kw_dk"
    if re.search(rf"\b({own})\b", c_noctx):
        return "yes", "kw_affirm"
    if re.search(rf"\b({other})\b", c_noctx) or NONE_RE.search(c_noctx):
        return "no", "kw_opposite"
    return "unparsed", "unparsed"


def normalise(df, qmeta=None):
    """Add Qx_norm and Qx_rule columns."""
    qmeta = QMETA if qmeta is None else qmeta
    out = df.copy()
    for q in qmeta.index:
        if (
            q + "_code" in out.columns
        ):  # answers already coded (e.g. symbols package incl. human review)
            out[q + "_norm"] = out[q + "_code"].map(CODE_MAP).fillna("unparsed")
            out[q + "_rule"] = "package_code"
            continue
        res = out[q].map(NORMALISERS[qmeta.loc[q, "target"]])
        out[q + "_norm"] = res.map(lambda r: r[0])
        out[q + "_rule"] = res.map(lambda r: r[1])
    return out


# ---------------------------------------------------------------------------
# Politicians dataset: "Is there a politician in the picture?" (target 'politician') and
# "Is any of Trump / Zelenskyy / Putin in the picture?" (target 'named')
# ---------------------------------------------------------------------------
POLITICIAN_WORDS = (
    r"politicians?|political (figure|leader)s?|president|prime minister|minister|chancellor|senator"
    r"|member of (the )?(european )?parliament|\bmp\b|mep|mayor|head of (state|government)"
    r"|statesman|congress(wo)?man|governor|party leader|lawmaker"
)
NON_POLITICIAN_WORDS = (
    r"coach|footballer|football (player|manager)|player|athlete|actor|actress|singer|musician"
    r"|journalist|news anchor|anchor|presenter|reporter|officer|soldier|celebrity|businessman"
    r"|businesswoman|entrepreneur|director|artist|model|influencer|doctor|scientist"
)
POL_DK = re.compile(
    r"^i (don'?t|do not) know|i (can'?t|cannot|am unable to|'m unable to|am not able to) (identify|determine|tell|say|recogni)"
    r"|(not|isn'?t) possible to (definitively |reliably |clearly |accurately )?(determine|identify|tell|say)|cannot (definitively |reliably |accurately )?(say|determine|identify|tell)"
    r"|not (clear|enough|sufficient)[^.]{0,30}(identify|determine)|does not provide (enough|sufficient)"
    r"|difficult to (determine|identify|say|tell)|unable to (determine|identify)|without (more |additional )?(context|information)"
)
NAMED_WORDS = r"trump|zelens\w*|zelenskyy|selenskyj|putin"


def _norm_yes_no_text(s, own_words, other_words, dk_re, none_of_listed=False):
    if _isna(s):
        return "missing", "missing"
    c = _clean(s)
    if c in YESNO_CATS:
        return c, "exact"
    if none_of_listed and re.search(
        r"none of the (mentioned|listed|following|named|three)", c
    ):
        return "no", "kw_none_of_listed"
    w = _first_word(c)
    if w in YESNO_CATS:
        return w, "first"
    a = _explicit_answer(c)
    if a in YESNO_CATS:
        return a, "answer_is"
    if re.search(r"\bhighly unlikely\b|\bunlikely that there is\b", c):
        return "no", "kw_negated"
    if dk_re.search(
        c
    ):  # hedging first: "it is not possible to determine if there is a politician" = dk, not no
        return "dk", "kw_dk"
    if re.search(rf"{NEGATOR}[^.;]{{0,40}}?\b({own_words})\b", c):
        return "no", "kw_negated"
    if re.search(rf"\b({own_words})\b", c):
        return "yes", "kw_affirm"
    if other_words and re.search(rf"\b({other_words})\b", c):
        return "no", "kw_opposite"
    return "unparsed", "unparsed"


def norm_politician(s):
    """'Is there a politician in the picture?' -> yes | no | dk | unparsed | missing."""
    return _norm_yes_no_text(s, POLITICIAN_WORDS, NON_POLITICIAN_WORDS, POL_DK)


def norm_named(s):
    """'Is Trump / Zelenskyy / Putin in the picture?' -> yes | no | dk | unparsed | missing."""
    return _norm_yes_no_text(s, NAMED_WORDS, None, POL_DK, none_of_listed=True)


# Symbols dataset: "Does the picture depict a national or EU symbol?" (target 'symbol')
SYMBOL_WORDS = r"flags?|emblems?|coat of arms|national symbols?|eu symbols?|european union symbols?"
CODE_MAP = {
    1: "yes",
    0: "no",
    9: "dk",
    8: "unclear",
}  # codes of the symbols reproducible package


def norm_symbol(s):
    """Fallback text rule for symbol questions (normally the package codes are used)."""
    return _norm_yes_no_text(s, SYMBOL_WORDS, None, POL_DK)


def norm_yesno_generic(s):
    """Generic yes/no question: exact / first word / 'the answer is ...' / hedging (dk); other free text is unparsed."""
    return _norm_yes_no_text(s, r"(?!x)x", None, POL_DK)


NORMALISERS = {
    "yesno": norm_yesno_generic,
    "symbol": norm_symbol,
    "polarity": norm_polarity,
    "negative": lambda s: norm_yesno(s, "negative"),
    "positive": lambda s: norm_yesno(s, "positive"),
    "politician": norm_politician,
    "named": norm_named,
}


def rule_counts(df):
    """How many answers were normalised by which rule (per model and question)."""
    t = pd.concat(
        {q: df.groupby("model")[q + "_rule"].value_counts() for q in QIDS},
        names=["qid"],
    )
    return t.unstack(fill_value=0)


def mapping_review(df, rules=None):
    """Distinct raw answers that are not exact categories, with their normalised value and rule."""
    rows = []
    for q in QIDS:
        sub = df[["model", q, q + "_norm", q + "_rule"]].dropna(subset=[q])
        sub = sub[~sub[q + "_rule"].isin(["exact", "missing"])]
        if rules is not None:
            sub = sub[sub[q + "_rule"].isin(rules)]
        sub.columns = ["model", "raw", "normalised", "rule"]
        sub.insert(1, "qid", q)
        rows.append(sub)
    r = pd.concat(rows)
    return (
        r.value_counts()
        .rename("n")
        .reset_index()
        .sort_values(["model", "qid", "rule", "normalised"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4. Scoring: paired test (exact McNemar) and Holm correction
# ---------------------------------------------------------------------------


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _annotate(ax, mat, fmt="{:.2f}", dark_above=None, dark_below=None, fontsize=8):
    for (i, j), v in np.ndenumerate(np.asarray(mat, dtype=float)):
        if np.isnan(v):
            continue
        white = (dark_above is not None and v > dark_above) or (
            dark_below is not None and v < dark_below
        )
        ax.text(
            j,
            i,
            fmt.format(v),
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white" if white else "black",
        )


def plot_heatmap(
    mat, title, ax, cmap="RdYlGn", vmin=0, vmax=1, xlabels=None, fontsize=8
):
    ax.imshow(
        np.asarray(mat, dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
    )
    _annotate(ax, mat, fontsize=fontsize, dark_below=0.6 if cmap == "viridis" else None)
    ax.set_xticks(range(mat.shape[1]), xlabels or list(mat.columns), fontsize=7)
    ax.set_yticks(range(mat.shape[0]), list(mat.index), fontsize=8)
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Simple question-accuracy analysis (one criterion: accuracy, DK = wrong)
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    # clamp so floating-point rounding at p=0 or p=1 cannot put p outside its own interval
    return max(0.0, min(p, c - h)), min(1.0, max(p, c + h))


def pairwise_question_tests(long, model):
    """Exact McNemar test for every pair of questions of one model (same images), Holm-corrected."""
    w = long[(long.model == model) & long.scored].pivot(
        index="filename", columns="qid", values="correct"
    )
    qids = [q for q in QIDS if q in w.columns]
    rows = []
    for i, a in enumerate(qids):
        for b in qids[i + 1 :]:
            d = w[[a, b]].dropna().astype(bool)
            only_a, only_b = int((d[a] & ~d[b]).sum()), int((~d[a] & d[b]).sum())
            rows.append(
                {
                    "model": model,
                    "q_a": a,
                    "q_b": b,
                    "n": len(d),
                    "acc_a": d[a].mean(),
                    "acc_b": d[b].mean(),
                    "only_a_correct": only_a,
                    "only_b_correct": only_b,
                    "p_value": mcnemar_exact(only_a, only_b),
                }
            )
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_value)
    t["significant"] = t.p_holm < 0.05
    return t


def plot_pairwise(tests, ax, title=""):
    """Heatmap of accuracy differences (row question minus column question); * = significant after Holm."""
    mat = pd.DataFrame(np.nan, index=QIDS, columns=QIDS)
    sig = pd.DataFrame(False, index=QIDS, columns=QIDS)
    for r in tests.itertuples():
        mat.loc[r.q_a, r.q_b], mat.loc[r.q_b, r.q_a] = (
            r.acc_a - r.acc_b,
            r.acc_b - r.acc_a,
        )
        sig.loc[r.q_a, r.q_b] = sig.loc[r.q_b, r.q_a] = r.significant
    lim = max(0.1, np.nanmax(np.abs(mat.to_numpy())))
    ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu", vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(mat.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(
                j,
                i,
                f"{v:+.2f}" + ("*" if sig.iat[i, j] else ""),
                ha="center",
                va="center",
                fontsize=7,
            )
    ax.set_xticks(range(len(QIDS)), QIDS)
    ax.set_yticks(
        range(len(QIDS)), [qlabel(q).replace("\n", " ") for q in QIDS], fontsize=8
    )
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Difficult images
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def image_index(*image_dirs):
    """Map file name -> path for all images below the given folder(s) (the first folder that exists wins)."""
    for d in image_dirs:
        d = Path(d).expanduser()
        if d.is_dir():
            return {
                p.name: p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
            }
    return {}


def difficulty_by_group(diff, by="subfolder", hard_threshold=0.5):
    g = diff.groupby(by)
    return pd.DataFrame(
        {
            "images": g.size(),
            "mean_share_wrong": g.share_wrong.mean(),
            f"hard_images (≥{hard_threshold:.0%} wrong)": g.share_wrong.apply(
                lambda x: int((x >= hard_threshold).sum())
            ),
        }
    ).sort_values("mean_share_wrong", ascending=False)


def _load_thumbnail(path, size=700):
    from PIL import Image

    img = Image.open(path)
    img.thumbnail((size, size))
    return img.convert("RGB")


def plot_image_evaluation(filename, long, images, title=""):
    """Image (left) and a grid of the evaluation (right): one row per model, one column per question.
    Cell = normalised answer; green = correct, red = wrong (DK counts as wrong), grey = missing / not scored."""
    g = long[long.filename == filename]
    models = list(dict.fromkeys(g.model))
    first = g.iloc[0]
    expected = g.drop_duplicates("qid").set_index("qid").truth.reindex(QIDS)

    fig, (ax_img, ax_tab) = plt.subplots(
        1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    path = images.get(filename)
    if path is not None:
        ax_img.imshow(_load_thumbnail(path))
    else:
        ax_img.text(0.5, 0.5, "image not found", ha="center", va="center")
    ax_img.axis("off")
    ax_img.set_title(
        f"{filename}\nlabel: {first.label}  |  group: {first.group}  |  folder: {first.subfolder}",
        fontsize=9,
    )

    rows = ["expected"] + models
    color = np.full(
        (len(rows), len(QIDS)), 0.5
    )  # 0 = wrong, 1 = correct, 0.5 = neutral
    text = [[("" if pd.isna(expected[q]) else expected[q]) for q in QIDS]]
    for i, m in enumerate(models, start=1):
        r = g[g.model == m].set_index("qid").reindex(QIDS)
        text.append(list(r.pred.fillna("")))
        for j, q in enumerate(QIDS):
            if r.scored.get(q, False):
                color[i, j] = float(r.correct[q])
    from matplotlib.colors import ListedColormap

    ax_tab.imshow(
        color,
        cmap=ListedColormap(["#f4a6a6", "#dddddd", "#a8dba8"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for i in range(len(rows)):
        for j in range(len(QIDS)):
            ax_tab.text(
                j,
                i,
                text[i][j],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if i == 0 else "normal",
            )
    ax_tab.set_xticks(range(len(QIDS)), [qlabel(q) for q in QIDS], fontsize=7)
    ax_tab.set_yticks(range(len(rows)), rows, fontsize=8)
    ax_tab.axhline(0.5, color="black", lw=1)
    n_s = g.scored.sum()
    ax_tab.set_title(
        f"{title}  wrong: {int(n_s - g.correct.sum())} of {int(n_s)} answers (green = correct, red = wrong)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def wrong_answers(filename, long, max_chars=160):
    """Raw answers behind the wrong evaluations of one image."""
    g = long[(long.filename == filename) & long.scored & (long.correct == 0)]
    out = g[["model", "qid", "truth", "pred", "raw"]].copy()
    out["raw"] = out.raw.astype(str).str.replace("\n", " ").str.slice(0, max_chars)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Expected answers, long table, accuracy
# ---------------------------------------------------------------------------
def to_long(df):
    """One row per model x image x question with the normalised answer (pred), the expected answer (truth),
    whether it is scored (an expected answer exists and the answer is not missing) and whether it is correct."""
    parts = []
    for q in QIDS:
        target, style, label_col = QMETA.loc[q, ["target", "style", "label_col"]]
        part = df[["model", "filename", "label", "group", "subfolder"]].copy()
        part["qid"], part["target"], part["style"] = q, target, style
        part["raw"] = df[q]
        part["pred"] = df[q + "_norm"]
        part["truth"] = df[label_col].astype(str).map(EXPECTED[target])
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["scored"] = long["truth"].notna() & (long["pred"] != "missing")
    long["correct"] = np.where(long["scored"], long["pred"] == long["truth"], np.nan)
    return long


def accuracy_table(long, by=("model", "qid")):
    """n, correct, accuracy with 95% Wilson CI, DK share and majority baseline for each group (scored rows only).
    majority_baseline = accuracy of always giving the most frequent expected answer."""
    rows = []
    for key, g in long[long.scored].groupby(list(by), sort=False):
        n, k = len(g), int(g.correct.sum())
        lo, hi = wilson_ci(k, n)
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update(
            n=n,
            correct=k,
            accuracy=k / n,
            ci_low=lo,
            ci_high=hi,
            dk_share=float((g.pred == "dk").mean()),
            majority_baseline=float(g.truth.value_counts(normalize=True).iloc[0]),
        )
        rows.append(r)
    return pd.DataFrame(rows)


def image_difficulty(long):
    """Per image: number of scored answers, number wrong and share wrong (over all models and questions)."""
    s = long[long.scored]
    d = (
        s.groupby(["filename", "label", "group", "subfolder"])["correct"]
        .agg(n_answers="count", n_correct="sum")
        .reset_index()
    )
    d["n_wrong"] = d.n_answers - d.n_correct
    d["share_wrong"] = d.n_wrong / d.n_answers
    return d.drop(columns="n_correct").sort_values(
        ["share_wrong", "filename"], ascending=[False, True]
    )


def pick_examples(diff, per_group=2, order=None):
    """The `per_group` hardest images of each image group."""
    order = order or sorted(diff.group.unique())
    return pd.concat(
        [
            diff[diff.group == g].head(per_group)
            for g in order
            if (diff.group == g).any()
        ]
    )


# ---------------------------------------------------------------------------
# Result sentences generated from the numbers
# ---------------------------------------------------------------------------
def show_result(text):
    display(Markdown(f"> **Result:** {text}"))


def _f(x):
    return f"{x:.2f}"


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def result_data(data, rules):
    n_img = data.filename.nunique()
    counts = data.drop_duplicates("filename").label.value_counts().to_dict()
    total = int(rules.sum().sum() - rules.get("missing", pd.Series(0)).sum())
    kw = int(
        rules.drop(
            columns=["exact", "first", "missing", "package_code"], errors="ignore"
        )
        .sum()
        .sum()
    )
    unparsed = int(rules.get("unparsed", pd.Series(0)).sum())
    coded = int(rules.get("package_code", pd.Series(0)).sum())
    missing = int(rules.get("missing", pd.Series(0)).sum())
    t = (
        f"{n_img} images have results and a ground-truth label "
        f"(label: number of images – {', '.join(f'{k}: {v}' for k, v in counts.items())}). "
    )
    if coded:
        t += f"{coded} answers are taken from the pre-coded results. "
    t += f"{kw} of {total} answers needed keyword rules and {unparsed} {'remains' if unparsed == 1 else 'remain'} unparsed"
    t += f"; {missing} answers are missing." if missing else "."
    return t


def result_accuracy(acc):
    a = acc.sort_values("accuracy", ascending=False)
    b, w = a.iloc[0], a.iloc[-1]
    t = (
        f"The most accurate question is {b.qid} ({_f(b.accuracy)}, 95% CI {_f(b.ci_low)}–{_f(b.ci_high)}), "
        f"the least accurate {w.qid} ({_f(w.accuracy)}). "
    )
    dk = acc[acc.dk_share >= 0.05]
    t += (
        f"DK answers matter for {_join(f'{r.qid} ({r.dk_share:.0%})' for r in dk.itertuples())}. "
        if len(dk)
        else "DK answers play no role (below 5% for every question). "
    )
    weak = acc[acc.accuracy <= acc.majority_baseline + 0.02]
    if len(weak):
        t += (
            f"{_join(weak.qid)} {'is' if len(weak) == 1 else 'are'} not better than always giving the most "
            f"frequent expected answer (baseline {_join(_f(x) for x in weak.majority_baseline)}). "
        )
    imb = acc[acc.majority_baseline >= 0.8]
    if len(imb):
        t += (
            f"For {_join(imb.qid)} the expected answers are imbalanced (baseline "
            f"{_join(_f(x) for x in imb.majority_baseline)}), so read the accuracy per group."
        )
    return t


def result_groups(acc_group):
    m = acc_group.drop(index="all images", errors="ignore")
    cells = m.stack().sort_values()
    low = cells.head(2)
    t = f"The weakest combinations are {_join(f'{g} with {q} ({_f(v)})' for (g, q), v in low.items())}. "
    good = [g for g in m.index if (m.loc[g] >= 0.95).all()]
    if good:
        t += f"{_join(good)} {'is' if len(good) == 1 else 'are'} answered correctly in at least 95% of cases with every question."
    return t


def result_pairs(tests):
    sig = tests[tests.significant].sort_values("p_holm")
    if not len(sig):
        return f"No difference between two questions is significant (smallest p_holm = {_f(tests.p_holm.min())})."
    items = [
        f"{r.q_a if r.acc_a > r.acc_b else r.q_b} > {r.q_b if r.acc_a > r.acc_b else r.q_a} "
        f"({_f(max(r.acc_a, r.acc_b))} vs. {_f(min(r.acc_a, r.acc_b))})"
        for r in sig.head(6).itertuples()
    ]
    more = f" and {len(sig) - 6} more" if len(sig) > 6 else ""
    return f"{len(sig)} of {len(tests)} question pairs differ significantly: {_join(items)}{more}."


def best_questions(acc, tests, questions=None):
    """Most accurate question and the questions not significantly worse than it (per model)."""
    rows = []
    for m, a in acc.groupby("model", sort=False):
        a = a[a.qid.isin(questions)] if questions else a
        a = a.sort_values("accuracy", ascending=False)
        best = a.iloc[0].qid
        t = tests[tests.model == m]
        tied = [best] + [
            q
            for q in a.qid[1:]
            if not t[
                ((t.q_a == best) & (t.q_b == q)) | ((t.q_a == q) & (t.q_b == best))
            ].significant.iloc[0]
        ]
        rows.append(
            {
                "model": m,
                "best_question": best,
                "best_accuracy": a.iloc[0].accuracy,
                "not_significantly_worse": ", ".join(sorted(tied)),
                "worst_question": a.iloc[-1].qid,
                "worst_accuracy": a.iloc[-1].accuracy,
                "mean_accuracy": a.accuracy.mean(),
            }
        )
    return pd.DataFrame(rows)


def result_best(best):
    if len(best) == 1:
        r = best.iloc[0]
        tied = r.not_significantly_worse.split(", ")
        t = f"Use {r.best_question} ({_f(r.best_accuracy)})"
        others = [q for q in tied if q != r.best_question]
        return t + (
            f"; {_join(others)} {'is' if len(others) == 1 else 'are'} not significantly worse, so the "
            f"simplest of these wordings is an equally good choice."
            if others
            else "; every other question is significantly worse."
        )
    per = _join(
        f"{r.model}: {r.best_question} ({_f(r.best_accuracy)})"
        for r in best.itertuples()
    )
    sets = [set(s.split(", ")) for s in best.not_significantly_worse]
    common = sorted(set.intersection(*sets))
    t = f"Best question per model – {per}. "
    t += (
        f"{_join(common)} {'is' if len(common) == 1 else 'are'} the best or not significantly worse than the best "
        f"for every model, i.e. a safe choice across models."
        if common
        else "No question is the best or not significantly worse than the best for all models, so the choice "
        "of wording depends on the model."
    )
    return t


def result_difficulty(diff):
    zero = int((diff.n_wrong == 0).sum())
    hard = diff[diff.share_wrong >= 0.5]
    g = diff.groupby("group").share_wrong.mean().sort_values()
    t = (
        f"{zero} of {len(diff)} images are answered correctly in all {int(diff.n_answers.max())} answers, and "
        f"{len(hard)} are wrong in half or more"
    )
    t += (
        f" ({', '.join(f'{v} {k}' for k, v in hard.group.value_counts().items())}). "
        if len(hard)
        else ". "
    )
    t += (
        f"The hardest group is {g.index[-1]} ({g.iloc[-1]:.0%} wrong answers on average), "
        f"the easiest {g.index[0]} ({g.iloc[0]:.0%})."
    )
    return t


def result_examples(examples, long):
    lines = []
    for r in examples.itertuples():
        w = long[(long.filename == r.filename) & long.scored & (long.correct == 0)]
        if not len(w):
            lines.append(f"`{r.filename}` ({r.group}): no wrong answers.")
            continue
        (truth, pred), n = (
            w.groupby(["truth", "pred"]).size().sort_values().index[-1],
            w.groupby(["truth", "pred"]).size().max(),
        )
        models = w.model.value_counts()
        who = f" by {_join(models.index)}" if long.model.nunique() > 1 else ""
        lines.append(
            f"`{r.filename}` ({r.group}): {int(r.n_wrong)} of {int(r.n_answers)} answers wrong{who}; "
            f'most often "{pred}" instead of "{truth}" ({n}x).'
        )
    return (
        "<br>".join(lines)
        + "<br>Look at the images: errors on every question point to a hard image or a questionable label, errors on some questions to the wording."
    )


def result_models_accuracy(acc):
    per = _join(
        f"{m} {_f(a.accuracy.min())}–{_f(a.accuracy.max())} (best {a.sort_values('accuracy').iloc[-1].qid})"
        for m, a in acc.groupby("model", sort=False)
    )
    b = acc.sort_values("accuracy").iloc[-1]
    t = f"Accuracy range per model: {per}. The best combination is {b.model} with {b.qid} ({_f(b.accuracy)}). "
    dk = acc[acc.dk_share >= 0.1].sort_values("dk_share", ascending=False)
    if len(dk):
        t += (
            "High DK shares lower the accuracy for "
            f"{_join(f'{r.model} {r.qid} ({r.dk_share:.0%})' for r in dk.head(5).itertuples())}. "
        )
    imb = acc.groupby("qid").majority_baseline.first()
    imb = imb[imb >= 0.8]
    if len(imb):
        t += (
            f"For {_join(imb.index)} the expected answers are imbalanced (baseline "
            f"{_join(_f(x) for x in imb)}), so a high accuracy there is easy to reach; read it per group."
        )
    return t


def result_models_lines(acc):
    s = acc.groupby("qid").accuracy.agg(lambda x: x.max() - x.min()).reindex(QIDS)
    return (
        f"The models agree most on {s.idxmin()} (spread {_f(s.min())}) and differ most on {s.idxmax()} "
        f"(spread {_f(s.max())} between the best and the worst model)."
    )


def result_models_groups(acc_group):
    items = []
    for m, g in acc_group.groupby(level=0, sort=False):
        cell = g.droplevel(0).stack().sort_values()
        (grp, q), v = cell.index[0], cell.iloc[0]
        items.append(f"{m}: {grp} with {q} ({_f(v)})")
    return f"Weakest group and question per model – {_join(items)}."


def result_models_pairs(tests):
    n = tests.groupby("model").size().iloc[0]
    s = tests.groupby("model", sort=False).significant.sum()
    return (
        f"Significant question differences (of {n} pairs): {_join(f'{m} {int(v)}' for m, v in s.items())}. "
        + (
            "No question difference is significant for any model."
            if s.sum() == 0
            else ""
        )
    )


# ---------------------------------------------------------------------------
# 5. Notebook sections: each shows its tables / plots / result sentence and saves them in run.out
# ---------------------------------------------------------------------------
def _save(fig, run, name):
    fig.tight_layout()
    fig.savefig(run.out / name, dpi=150)
    plt.show()


def show_questions(run):
    """The questions and how their expected answer is derived from the ground truth."""
    t = pd.DataFrame(
        {
            "type": QMETA.target,
            "style": QMETA["style"],
            "question": QMETA.question,
            "expected answer (label -> answer)": [
                (f"{c}: " if c != "label" else "")
                + ", ".join(f"{k} -> {v}" for k, v in EXPECTED[t].items())
                for t, c in zip(QMETA.target, QMETA.label_col)
            ],
        }
    )
    display(t)


def show_normalisation(run):
    """Which rules turned the raw answers into categories, and a sample of the keyword-mapped answers."""
    data = run.data[run.data.filename.isin(run.long.filename)]
    rules = rule_counts(data)
    display(rules)
    review = mapping_review(
        data,
        rules=[
            r
            for r in rules.columns
            if r not in ("exact", "first", "missing", "package_code")
        ],
    )
    review.to_csv(run.out / "mapping_review.csv", index=False)
    if len(review):
        print(
            f"{len(review)} distinct keyword-mapped or unparsed answers (all in {run.out / 'mapping_review.csv'}); sample:"
        )
        display(
            pd.concat(
                [
                    review[review.rule == "unparsed"],
                    review[review.rule != "unparsed"]
                    .sample(frac=1, random_state=0)
                    .groupby(["model", "rule"])
                    .head(3),
                ]
            )[["model", "qid", "rule", "normalised", "raw"]]
        )
    text = result_data(data, rules)
    if run.n_dropped:
        text += f" {run.n_dropped} images are dropped because a model has no answer for them."
    show_result(text)


def show_accuracy(run):
    """Accuracy per (model and) question, with 95% CI, DK share and majority baseline."""
    acc = accuracy_table(run.long).merge(
        QMETA[["target", "style"]], left_on="qid", right_index=True
    )
    acc.to_csv(run.out / "accuracy.csv", index=False)
    xl = [qlabel(q) for q in QIDS]
    if len(run.models) == 1:
        display(
            acc[
                [
                    "qid",
                    "target",
                    "style",
                    "n",
                    "correct",
                    "accuracy",
                    "ci_low",
                    "ci_high",
                    "dk_share",
                    "majority_baseline",
                ]
            ].round(3)
        )
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(acc)), 4))
        x = np.arange(len(acc))
        palette = dict(zip(QMETA.target.unique(), plt.cm.tab10.colors))
        ax.bar(
            x,
            acc.accuracy,
            color=acc.target.map(palette),
            yerr=np.clip(
                [acc.accuracy - acc.ci_low, acc.ci_high - acc.accuracy], 0, None
            ),
            capsize=4,
        )
        ax.scatter(
            x,
            acc.majority_baseline,
            marker="_",
            s=600,
            color="black",
            zorder=3,
            label="majority baseline",
        )
        for xi, v in zip(x, acc.accuracy):
            ax.text(xi, 0.03, f"{v:.2f}", ha="center", color="white", fontsize=9)
        ax.set_xticks(x, [qlabel(q) for q in acc.qid], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Accuracy per question (95% Wilson CI); colour = question type")
        _save(fig, run, "accuracy.png")
        show_result(result_accuracy(acc))
        return acc

    acc_mat = acc.pivot(index="model", columns="qid", values="accuracy").reindex(
        index=run.models, columns=QIDS
    )
    dk_mat = acc.pivot(index="model", columns="qid", values="dk_share").reindex(
        index=run.models, columns=QIDS
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(max(12, 2.2 * len(QIDS) + 4), 0.6 * len(run.models) + 1.5)
    )
    plot_heatmap(acc_mat, "Accuracy (DK counts as wrong)", axes[0], xlabels=xl)
    plot_heatmap(dk_mat, "Share of DK answers", axes[1], cmap="Reds", xlabels=xl)
    _save(fig, run, "accuracy_heatmap.png")
    display(
        acc.groupby("qid").majority_baseline.first().reindex(QIDS).round(3).to_frame().T
    )

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(QIDS)), 4.5))
    x = np.arange(len(QIDS))
    for i, m in enumerate(run.models):
        s = acc[acc.model == m].set_index("qid").reindex(QIDS)
        ax.errorbar(
            x + (i - (len(run.models) - 1) / 2) * 0.08,
            s.accuracy,
            yerr=np.clip([s.accuracy - s.ci_low, s.ci_high - s.accuracy], 0, None),
            marker="o",
            capsize=3,
            label=m,
        )
    ax.set_xticks(x, xl, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Accuracy per question and model (95% Wilson CI)")
    _save(fig, run, "accuracy_lines.png")
    show_result(result_models_accuracy(acc) + " " + result_models_lines(acc))
    return acc


def show_groups(run, acc):
    """Accuracy per image group (and model) and question."""
    by = ("qid", "group") if len(run.models) == 1 else ("model", "group", "qid")
    t = accuracy_table(run.long, by=by)
    if len(run.models) == 1:
        mat = t.pivot(index="group", columns="qid", values="accuracy").reindex(
            columns=QIDS
        )
        mat.loc["all images"] = acc.set_index("qid").accuracy
        labels = list(mat.index)
    else:
        mat = t.pivot(
            index=["model", "group"], columns="qid", values="accuracy"
        ).reindex(columns=QIDS)
        labels = [f"{m} – {g}" for m, g in mat.index]
    mat.round(2).to_csv(run.out / "accuracy_per_group.csv")
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(QIDS) + 3), 0.45 * len(mat) + 1.5))
    plot_heatmap(
        mat,
        "Accuracy per image group and question",
        ax,
        xlabels=[qlabel(q) for q in QIDS],
    )
    ax.set_yticks(range(len(mat)), labels, fontsize=8)
    _save(fig, run, "accuracy_per_group.png")
    show_result(
        result_groups(mat) if len(run.models) == 1 else result_models_groups(mat)
    )


def show_pairs(run):
    """Exact McNemar test for every pair of questions (per model), Holm-corrected."""
    tests = pd.concat(
        [pairwise_question_tests(run.long, m) for m in run.models], ignore_index=True
    )
    tests.to_csv(run.out / "question_pairs_mcnemar.csv", index=False)
    ncol = min(2, len(run.models))
    nrow = int(np.ceil(len(run.models) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * (0.9 * len(QIDS) + 4), nrow * (0.8 * len(QIDS) + 1.5)),
        squeeze=False,
    )
    for ax, m in zip(axes.flat, run.models):
        plot_pairwise(
            tests[tests.model == m], ax, f"{m}: row minus column; * = significant"
        )
    for ax in list(axes.flat)[len(run.models) :]:
        ax.axis("off")
    _save(fig, run, "question_pairs.png")
    if len(run.models) == 1:
        display(
            tests[
                [
                    "q_a",
                    "q_b",
                    "acc_a",
                    "acc_b",
                    "only_a_correct",
                    "only_b_correct",
                    "p_holm",
                    "significant",
                ]
            ].round(4)
        )
        show_result(result_pairs(tests))
    else:
        show_result(result_models_pairs(tests))
    return tests


def show_best(run, acc, tests):
    """Most accurate question (per model) and the questions not significantly worse than it."""
    best = best_questions(acc, tests, run.cfg.get("summary_questions"))
    best.to_csv(run.out / "best_questions.csv", index=False)
    display(best.round(3))
    show_result(result_best(best))


def show_difficult_images(run, per_group=2):
    """Histogram of wrong answers per image, difficulty per group, and the hardest images of each group."""
    long = run.long
    diff = image_difficulty(long)
    if len(run.models) > 1:
        per_model = (
            long[long.scored]
            .assign(wrong=lambda d: 1 - d.correct)
            .pivot_table(
                index="filename", columns="model", values="wrong", aggfunc="sum"
            )
            .reindex(columns=run.models)
        )
        diff = diff.join(per_model, on="filename")
    diff.to_csv(run.out / "image_difficulty.csv", index=False)

    n_max = int(diff.n_answers.max())
    fig, ax = plt.subplots(figsize=(min(12, 4 + 0.3 * n_max), 3))
    counts = diff.n_wrong.value_counts().reindex(range(n_max + 1), fill_value=0)
    ax.bar(counts.index, counts.values, color="tab:gray")
    ax.set_xlabel(f"wrong answers per image (of {n_max})")
    ax.set_ylabel("images")
    ax.set_title("How many answers go wrong per image")
    _save(fig, run, "image_difficulty_hist.png")
    display(difficulty_by_group(diff, by=["group", "subfolder"]).round(2))
    show_result(result_difficulty(diff))

    examples = pick_examples(
        diff, per_group=per_group, order=run.cfg.get("group_order")
    )
    display(
        Markdown(
            "**Hardest images per group** – left: the image; right: expected answer (first row) and "
            "normalised answers, green = correct, red = wrong. Below: the raw wrong answers."
        )
    )
    for fn in examples.filename:
        plot_image_evaluation(fn, long, run.images)
        plt.show()
        display(wrong_answers(fn, long))
    show_result(result_examples(examples, long))


def show_hardest_images(run, n=6):
    """The `n` images with the largest share of wrong answers: the image, every generated answer with its
    expected answer, and which answers are wrong."""
    diff = image_difficulty(run.long)
    diff.to_csv(run.out / "image_difficulty.csv", index=False)
    hardest = diff.sort_values(
        ["share_wrong", "n_wrong", "filename"], ascending=[False, False, True]
    ).head(n)
    print(
        f"{int((diff.share_wrong >= hardest.share_wrong.min()).sum())} of {len(diff)} scored images have at least "
        f"{hardest.share_wrong.min():.0%} wrong answers; the {len(hardest)} hardest are shown "
        f"(all in {run.out / 'image_difficulty.csv'})."
    )
    for r in hardest.itertuples():
        display(
            Markdown(
                f"### `{r.filename}`  \nlabel: **{r.label}**  |  group: {r.group}  |  folder: {r.subfolder}"
                f"  |  **{int(r.n_wrong)} of {int(r.n_answers)} answers wrong**"
            )
        )
        _fig, ax = plt.subplots(figsize=(6, 4.5))
        path = run.images.get(r.filename)
        if path is not None:
            ax.imshow(_load_thumbnail(path))
        else:
            ax.text(0.5, 0.5, "image not found in IMAGE_DIR", ha="center", va="center")
        ax.axis("off")
        plt.show()

        g = run.long[run.long.filename == r.filename].set_index("qid").reindex(QIDS)
        verdict = np.where(
            ~g.scored.astype(bool),
            "– not scored",
            np.where(g.correct == 1, "✓ correct", "✗ wrong"),
        )
        display(
            pd.DataFrame(
                {
                    "question": [qlabel(q).split("\n")[1] for q in QIDS],
                    "generated answer": g.raw.fillna("(no answer)")
                    .astype(str)
                    .str.replace("\n", " "),
                    "read as": g.pred,
                    "expected": g.truth.fillna("–"),
                    "": verdict,
                },
                index=QIDS,
            )
        )
        wrong = g[g.scored.astype(bool) & (g.correct == 0)]
        show_result(
            "Wrong answers: "
            + _join(
                f'{q} ("{w.pred}" instead of "{w.truth}")' for q, w in wrong.iterrows()
            )
            + "."
            if len(wrong)
            else "No wrong answers."
        )
