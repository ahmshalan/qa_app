from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from typing import List


class OpenAi:
    def __init__(
        self, model: str = "gpt-3.5-turbo", temperature: float = 0, **kwargs
    ):

        if self._is_chat(model):
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                model_kwargs={"seed": 235},
                api_key="sk-proj-PBcth1ldArSAsEc9dEaVT3BlbkFJF1IEpPv8xLHP183UWQod",
                # max_tokens=32000,
                # max_retries=10,
                **kwargs
            )
        else:
            self.llm = OpenAI(model=model, temperature=temperature, **kwargs)
        self.parser = JsonOutputParser()

    def _is_chat(self, model: str):
        chat_models = ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        if model in chat_models:
            return True
        return False

    async def abatch(self, prompt: str, payloads: List[dict]):
        prompt_template = PromptTemplate.from_template(prompt)
        llm_chain = prompt_template | self.llm | self.parser

        return await llm_chain.abatch(payloads, config={"max_concurrency": 10})  #
