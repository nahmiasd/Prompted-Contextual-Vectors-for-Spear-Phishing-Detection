import joblib
import pandas as pd
from src.constants import CACHE_DIR
from src.llms import get_gpt4, get_gemini_pro, get_gpt35
from src.chains import ask_question_chain_with_float_output_parser, ask_question_chain_with_pydantic_output_parser
import asyncio
from langchain_core.runnables import Runnable
from typing import List, Coroutine, Dict, Any
from tqdm import tqdm
from more_itertools import chunked
from src.utils import load_questions, load_emails, output_results, create_or_load_cache
from openai.error import InvalidRequestError
from argparse import ArgumentParser

LLMS = [get_gpt4(), get_gpt35(), get_gemini_pro()]  # fill this list with your own LLMs
LLM_NAMES = ['GPT4_1106', 'GPT35_0613', 'Gemini Pro']  # fill this list with the corresponding LLM names
CACHE_FILE_NAME = "async_cache.pickle"
CACHE = create_or_load_cache(CACHE_FILE_NAME)
MISS_FLAG = False


async def run_pipeline(batch_size: int, sleep_time: float):
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    print(questions)
    emails = load_emails()
    print(f"Loaded {len(emails)} emails")
    jobs = gather_async_jobs(emails, questions)
    results = await run_jobs_chunked(jobs, batch_size, sleep_time)
    output_results(results, emails, questions)


async def run_jobs_chunked(jobs: List[Coroutine], chunk_size: int, sleep_time: float) -> List[dict]:
    global CACHE, CACHE_FILE_NAME, MISS_FLAG
    initial_cache_len = len(CACHE)
    chunked_jobs = list(chunked(jobs, n=chunk_size))
    results = []
    for chunk in tqdm(chunked_jobs, desc="processing chunks...", total=len(chunked_jobs)):
        try:
            chunk_results = await asyncio.gather(*chunk)
            results += chunk_results
            if MISS_FLAG:  # In case we do call the LLM, we want to delay the next call so we will not exceed the
                # tokens per minute limit.
                await asyncio.sleep(sleep_time)
                MISS_FLAG = False
        except Exception as e:
            raise e
        finally:
            cache_path = CACHE_DIR / CACHE_FILE_NAME
            if not cache_path.is_file() or len(CACHE) > initial_cache_len:
                joblib.dump(CACHE, cache_path)
    return results


def gather_async_jobs(emails: pd.DataFrame, questions: List[str]) -> List[Coroutine]:
    jobs = []
    for email_hash, email in tqdm(zip(emails['email_hash'].tolist(), emails['email_str'].tolist()),
                                  desc="Gathering jobs...",
                                  total=emails.shape[0]):
        for llm, llm_name in zip(LLMS, LLM_NAMES):
            chain = ask_question_chain_with_pydantic_output_parser(llm)
            fallback_chain = ask_question_chain_with_float_output_parser(llm)
            chain = chain.with_fallbacks([fallback_chain])
            for question_id, question in enumerate(questions):
                prompt_input = {'question': question, 'email': email}
                job = call_llm(chain, email_hash, question_id, llm_name, prompt_input)
                jobs.append(job)
    return jobs


async def call_llm(chain: Runnable, email_hash: str, question_id: int, llm_name: str,
                   prompt_input: dict) -> Dict[str, Any]:
    """
    Cached calling for LLM with the input question
    """
    global CACHE, MISS_FLAG
    if (email_hash, question_id, llm_name) in CACHE:
        answer = CACHE[(email_hash, question_id, llm_name)]
    else:
        try:
            answer = await chain.ainvoke(prompt_input)
        except InvalidRequestError as e:  # if the LLM fails for known reasons, impute this response with 0.5.
            if "maximum context length" in str(e) or 'management policy' in str(e):
                answer = 0.5
            else:  # Otherwise, raise the exception
                raise e
        CACHE[(email_hash, question_id, llm_name)] = answer
        MISS_FLAG = True
    result = {'email_hash': email_hash, 'llm': llm_name, 'question_id': question_id, 'answer': answer}
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=60, help='Number of emails to process in parallel')
    parser.add_argument('--sleep_time', '-s', type=float, default=60, help='Time in seconds to sleep between batches')
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.batch_size, args.sleep_time))
