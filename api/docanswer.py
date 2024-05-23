__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import concurrent.futures
from typing import Optional, List
from langchain import hub
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


import json
import pymupdf as fitz  # PyMuPDF

from .utils import OpenAi


prompt = hub.pull("hwchase17/react")



class Response(BaseModel):
    """Final response to the question being asked."""

    answer: str = Field(description="The final answer to respond to the user")
    pageno: int = Field(
        description="The first page used to get the exact match, if there is no make it as 0"
    )
    reftext: str = Field(
        description="The exact match text that helps in getting the answer. Make sure to use only one context/sentence to get the answer and extract the info from. if there is no make it as empty string"
    )




def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )


class DocAnswer:
    db = None

    def __init__(
        self,
        filename: str,
        prompt: Optional[str] = None,
        model: Optional[str] = "gpt-3.5-turbo",
    ):
        self.filename = filename
        DocAnswer.set_db(self.embed())
        self.model = model

        if not prompt:
            self.prompt = """Using the input text, and list of instructions , generate a question that will help answering those instructions , then output json where it is having three values the answer, pageno , and the reftext which is the exact match text that helps in getting the answer.
                        Answer the last question only
                        If there are no reftext found make it as empty string.
                        Question: {question}
                        Input Text: {input_text}
                        Json Answer:"""
        else:
            self.prompt = prompt

    @classmethod
    def set_db(cls, db):
        cls.db = db

    @staticmethod
    @tool
    def search(query: str) -> List:
        """Look up things in document."""
        top_results = DocAnswer.db.similarity_search(query, k=3)

        return top_results

    def _get_text_coordinates(self, target_text, pageNo):
        try:
            # Open the PDF file
            pdf_document = fitz.open(self.filename)

            # Initialize an empty list to store coordinates
            coordinates = []
            page_width_inch, page_height_inch = 0, 0
            # Iterate through each page of the PDF

            page = pdf_document.load_page(pageNo)
            # Search for the target text on the page
            text_instances = page.search_for(target_text)

            # Remove sentences in reverse to try to get the sentence

            if not text_instances:
                for i in range(len(target_text.split("."))):
                    _target_text = ".".join(target_text.split(".")[: -i + 1])
                    text_instances = page.search_for(_target_text)
                    if text_instances:
                        break

            # Get page Width
            page_width_inch = page.rect.width / 72  # 1 inch = 72 points
            page_height_inch = page.rect.height / 72  # 1 inch = 72 points
            # Iterate through each instance of the target text
            for inst in text_instances:
                # Get the coordinates of the bounding box and append them to the list
                coordinates.append(inst)

            # Close the PDF document
            pdf_document.close()

            # Convert coordinates to four arrays
            num_instances = len(coordinates)

            def extract_corner_points(rect_list):
                corner_points = []
                x1, y1, x2, y2 = rect_list
                # Top-left corner
                corner_points.append([x1 / 72, y1 / 72])
                # Top-right corner
                corner_points.append([x2 / 72, y1 / 72])
                # Bottom-right corner
                corner_points.append([x2 / 72, y2 / 72])
                # Bottom-left corner
                corner_points.append([x1 / 72, y2 / 72])
                return corner_points

            if num_instances >= 1:
                formatted_coordinates = extract_corner_points(coordinates[0])

                return {
                    "contextloc": str(formatted_coordinates),
                    "pagedimension": f"Page has width: {page_width_inch} and height: {page_height_inch}, measured with unit: inch",
                }
            else:
                return None
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None

    def get_text_coordinates(self, inputs: List[dict]):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the task to the executor
            futures = [
                executor.submit(self._get_text_coordinates, **input) for input in inputs
            ]

            # Get the result from the future
            coords = [future.result() for future in futures]

        return coords

    def embed(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            # Set a really small chunk size, just to show.
            chunk_size=600,
            chunk_overlap=20,
        )

        chunks = UnstructuredFileLoader(self.filename, mode="paged").load_and_split(
            text_splitter
        )
        for chunk in chunks:
            chunk.metadata.pop("coordinates")
            chunk.metadata.pop("languages")

        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        db = Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key="sk-proj-PBcth1ldArSAsEc9dEaVT3BlbkFJF1IEpPv8xLHP183UWQod",
            ),
        )

        return db

    def _get_agent(self):

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant, help on extracting answers using the given context , try to make the answer short 1-3 words max, and follow the answer regex",
                ),
                ("user", "{input}\nAnswer Regex: {answer_regex}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        chain = OpenAi(self.model)
        llm_with_tools = chain.llm.bind_functions([self.search, Response])
        agent = (
            {
                "input": lambda x: x["input"],
                # Format agent scratchpad from intermediate steps
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "answer_regex": lambda x: x["answer_regex"],
            }
            | prompt
            | llm_with_tools
            | parse
        )

        # agent = create_react_agent(chain.llm, [self.search, Response], prompt)
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.search],
            # handle_parsing_errors=True,
            # verbose=True,
        )

        return agent_executor

    def _output_response(self, questions, mapping, answers, results):

        responses = []

        for i, answer in enumerate(answers):
            response = {}
            map_item = list(
                filter(
                    lambda item: str(item["questionId"]) == questions[i]["quesID"],
                    mapping,
                )
            )[0]
            response["err"] = "N"
            response["ans"] = answer["answer"] if not answer.get("output") else ""
            response["docid"] = map_item["documentId"]
            response["quesID"] = str(questions[i]["quesID"])
            response["reference"] = map_item["questionTextLocation"]
            response["errorCode"] = 0
            response["pagedimension"] = (
                results[i]["pagedimension"] if results[i] else ""
            )
            response["pageno"] = (
                str(answer["pageno"]) if not answer.get("output") else ""
            )
            response["reftext"] = (
                str(answer["reftext"]) if not answer.get("output") else ""
            )
            response["contextloc"] = results[i]["contextloc"] if results[i] else ""

            responses.append(response)

        return responses

    def parse_answers(self, answer):
        try:
            answer = {
                "target_text": answer["reftext"],
                "pageNo": (answer["pageno"] - 1),
            }
        except Exception as e:
            print(f"Error while parsing: {answer}")
            answer = {"target_text": "", "pageNo": 0}

        return answer

    async def answer_file(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Loading questions to answer
            with open("./Request_Json testing900load_id_400.json", "r") as f:
                questions = json.load(f)
                questions = [
                    question for question in questions["request"]["data"]["questions"]
                ]

            # Loading mapping
            with open("./QnAMapping.json", "rb") as f:
                mapping = json.load(f)

            chain = self._get_agent()

            answer_regex = lambda question: [
                item["answerRegex"]
                for item in mapping
                if str(item["questionId"]) == question["quesID"]
            ]
            answers = await chain.abatch(
                [
                    {
                        "input": question["question"],
                        "answer_regex": answer_regex(question)[0],
                    }
                    for question in questions
                ],
                config={"max_concurrency": 7},
                return_only_outputs=True,
            )


            inputs = list(
                map(
                    lambda answer: self.parse_answers(answer),
                    answers,
                )
            )

            results = self.get_text_coordinates(inputs)

            response = self._output_response(questions, mapping, answers, results)

            return response
