import re
from typing import Union

from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class FloatOutputParser(BaseOutputParser):
    def _get_last_float_from_text(self, llm_output: str) -> Union[float, str]:
        """
        Will get the last float from the output. We want the last float since LLMs usually output final answers
        in the end of the text. This way we avoid numbers that may occur in the reasoning steps of the LLM.
        :param llm_output: Output text of the llm
        :return: Last float found in the text
        """
        pattern = r'[-+]?\d*\.\d+|\d+'
        matches = re.findall(pattern, llm_output)
        if matches:
            matches = [float(x) for x in matches]
            for element in reversed(matches):
                if 0 < element < 1:
                    return element
            return float(matches[-1])
        else:
            return llm_output

    def parse(self, text: str) -> Union[float, str]:
        return self._get_last_float_from_text(text)


class CorrectionResultsModel(BaseModel):
    thoughts: str = Field(description="Your thoughts before coming up with the final answer")
    final_answer: str = Field(
        description="Your final answer as either python floating point number or 'No probability'")


class QuestionResultsModel(BaseModel):
    thoughts: str = Field(description="Summary of your thoughts before coming up with the final answer")
    final_answer: float = Field(
        description="Your final answer as a python floating point number. If you are not sure, use 0.5 here.")


CORRECTION_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=CorrectionResultsModel)
QUESTION_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=QuestionResultsModel)
