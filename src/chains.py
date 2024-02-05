from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from src.constants import PROMPTS_DIR
from langchain.chat_models.base import BaseChatModel
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from src.output_parsers import FloatOutputParser, QUESTION_OUTPUT_PARSER, CORRECTION_OUTPUT_PARSER
from src.prompts import ASK_QUESTION_PROMPT, PROBABILITY_COT_SYSTEM_MESSAGE


def ask_question_chain_with_float_output_parser(llm: BaseChatModel) -> Runnable:
    system_message = PROBABILITY_COT_SYSTEM_MESSAGE
    prompt = ChatPromptTemplate.from_messages(
        messages=[system_message, ASK_QUESTION_PROMPT])
    chain = prompt | llm | FloatOutputParser()
    return chain


def ask_question_chain_with_pydantic_output_parser(llm: BaseChatModel) -> Runnable:
    output_parser = QUESTION_OUTPUT_PARSER
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate.from_file(PROMPTS_DIR / "system_prompt_structured_output_CoT.txt",
                                      input_variables=['question', 'email'],
                                      partial_variables={"format": format_instructions})
    chain = prompt | llm | output_parser | RunnableLambda(lambda x: x.final_answer)
    return chain


def correct_result_chain(llm: BaseChatModel) -> Runnable:
    output_parser = CORRECTION_OUTPUT_PARSER
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate.from_file(PROMPTS_DIR / "result_correction_prompt.txt", input_variables=['text'],
                                      partial_variables={"format": format_instructions})
    chain = {'text': RunnablePassthrough()} | prompt | llm | output_parser
    return chain
