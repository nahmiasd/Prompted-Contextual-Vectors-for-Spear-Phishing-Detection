import pandas as pd
from src.chains import correct_result_chain
from src.llms import get_gpt4  # Or any other LLM
from typing import Any
from tqdm import tqdm
from langchain_core.runnables import Runnable
from pathlib import Path
from src.utils import create_or_load_cache
import joblib
from src.constants import CACHE_DIR
from langchain_core.exceptions import OutputParserException
from json.decoder import JSONDecodeError
import argparse
from langchain_core.language_models.chat_models import BaseChatModel

tqdm.pandas()

CACHE_FILENAME = "correction_cache.pkl"
CACHE = create_or_load_cache(CACHE_FILENAME)


def correct_result(correction_chain: Runnable, result: Any) -> float:
    try:
        result_float = float(result)
        return handle_float_case(result_float)
    except ValueError:
        return correct_with_llm(correction_chain, result)


def correct_with_llm(correction_chain: Runnable, result: Any) -> float:
    if result in CACHE:
        return CACHE[result]
    try:
        result_corrected_dict = correction_chain.invoke(result)
        result_corrected_answer = float(result_corrected_dict.final_answer)
    except (OutputParserException, JSONDecodeError, ValueError):  # Unable to correct the LLM response
        CACHE[result] = 0.5
        joblib.dump(CACHE, CACHE_DIR / CACHE_FILENAME)
        return 0.5
    CACHE[result] = result_corrected_answer
    joblib.dump(CACHE, CACHE_DIR / CACHE_FILENAME)
    return result_corrected_answer


def handle_float_case(result_float: float) -> float:
    if result_float > 1:
        return result_float / 100
    else:
        return result_float


def run_pipeline(results_df_path: Path, llm: BaseChatModel = get_gpt4()):
    results_df = pd.read_pickle(results_df_path)
    if 'email_hash' in results_df.columns:
        results_df = results_df.set_index('email_hash')
    for column in [col for col in results_df.columns if 'email' not in col]:
        correction_chain = correct_result_chain(llm=llm)
        results_df[column] = results_df[column].progress_apply(lambda x: correct_result(correction_chain, x))
    out_path = results_df_path.parent / 'results_corrected.pkl'
    results_df.to_pickle(out_path)
    print(f"Correction pipeline completed, results saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Path to the results DataFrame pickle file')
    args = parser.parse_args()
    input_df_path = Path(args.i)
    run_pipeline(input_df_path)
