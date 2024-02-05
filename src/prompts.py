from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.constants import PROMPTS_DIR

PROBABILITY_COT_SYSTEM_MESSAGE = SystemMessagePromptTemplate.from_template_file(
    PROMPTS_DIR.joinpath("system_prompt_probability_output_CoT.txt"), input_variables=[])
ASK_QUESTION_PROMPT = HumanMessagePromptTemplate.from_template_file(PROMPTS_DIR / "question_prompt.txt",
                                                                    input_variables=['question', 'text'])

