import hashlib
import json
import os
import re
from datetime import datetime
from glob import glob
from json import JSONDecodeError
from pathlib import Path
from typing import Union, List

import joblib

from src.constants import EMAILS_DIR
import pandas as pd
from tqdm import tqdm

from src.constants import QUESTIONS_FILE_PATH, EMAIL_SUBJECT_FIELD, EMAIL_BODY_FIELD, OUTPUT_DIR, CACHE_DIR


def load_questions() -> List[str]:
    with open(QUESTIONS_FILE_PATH, "r") as f:
        questions = [x.strip() for x in f.readlines()]
    return questions


def load_emails(root_dir: Union[str, Path] = EMAILS_DIR) -> pd.DataFrame:
    email_files = glob(os.path.join(root_dir, "**/*.json"), recursive=True)
    emails = []
    errors = 0
    for i, json_path in tqdm(enumerate(email_files), desc="Collecting emails json files", total=len(email_files)):
        try:
            j = json.load(open(json_path))
        except JSONDecodeError:
            errors += 1
            continue
        label = get_email_label_from_path(json_path)
        email_subject = (j.get(EMAIL_SUBJECT_FIELD, '') + j.get('subject', '')).strip()
        email_body = (j.get(EMAIL_BODY_FIELD, '') + j.get('email_body', '') + j.get('body', '')).strip()
        email_str = "\n".join([email_subject, email_body])
        email_hash = hashlib.md5(email_str.encode()).hexdigest()
        emails.append((i, email_str, email_hash, label))
    df = pd.DataFrame.from_records(emails, columns=["email_id", "email_str", "email_hash", "label"])
    df = df.drop_duplicates(subset=["email_hash"], keep="last")
    return df


def get_email_label_from_path(json_path: str) -> str:
    path_obj = Path(json_path)
    try:
        index = path_obj.parts.index('emails')
        folder_after_emails = path_obj.parts[index + 1]
        return folder_after_emails
    except ValueError:
        return 'no label found'


def create_output_df(results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(results)
    df['combined_column'] = df['llm'] + '_' + df['question_id'].astype(str)
    df = df.pivot_table(index='email_hash', columns=['combined_column'], values='answer', aggfunc='first')
    df = df.reset_index()
    return df


def output_results(results: List[dict], emails_df: pd.DataFrame, questions: List[str],
                   results_filename: str = 'results'):
    output_path = OUTPUT_DIR / f"{get_now_date()}"
    output_path.mkdir(parents=True, exist_ok=True)
    results_df = create_output_df(results)
    results_df.to_pickle(output_path / f'{results_filename}.pkl')
    emails_df.to_pickle(output_path / f'emails.pkl')
    pd.DataFrame({'question_id': list(range(len(questions))), 'question': questions}).to_pickle(
        output_path / f'questions.pkl')
    print(f"Saved results to {output_path}")


def create_cache_dir(cache_dir: Union[str, Path] = CACHE_DIR) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_now_date() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M")


def create_or_load_cache(cache_file_name: str) -> dict:
    cache_dir = create_cache_dir()
    cache_file = cache_dir / cache_file_name
    if cache_file.exists():
        cache = joblib.load(cache_file)
        print(f"loaded cache from {cache_file}, cache length: {len(cache)}")
        return cache
    return {}


def preprocess_email_text(email_text: str) -> str:
    pattern = re.compile(r'\b\w+\b|[!.,<>#=]')
    matches = pattern.findall(email_text)
    result = ' '.join(matches)
    return result
