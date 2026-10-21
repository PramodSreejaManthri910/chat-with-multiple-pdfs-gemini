import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class CustomGeminiLLM(DeepEvalBaseLLM):
    """Custom Gemini model used as the DeepEval judge."""

    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = "gemini-2.5-flash"

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )

        return schema.model_validate_json(response.text)

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 2.5 Flash"


# Initialize Gemini evaluation model
custom_llm = CustomGeminiLLM()


# Initialize DeepEval metrics
contextual_precision = ContextualPrecisionMetric(model=custom_llm)
contextual_recall = ContextualRecallMetric(model=custom_llm)
contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)
answer_relevancy = AnswerRelevancyMetric(model=custom_llm)
faithfulness = FaithfulnessMetric(model=custom_llm)


# Sample RAG test case
test_case = LLMTestCase(
    input="Explain Types of Operating Systems",

    actual_output="""
    1. Batch OS - A set of similar jobs are stored in main memory
    for execution.

    2. Multiprogramming OS - Multiple jobs wait for CPU time,
    allowing the CPU to switch when a process is waiting.

    3. Multitasking OS - Uses rapid switching between jobs so
    multiple tasks appear to run simultaneously.

    4. Time Sharing OS - Allows users to interact with the system
    while CPU time is shared between tasks.

    5. Real Time OS - Designed for dedicated systems that must
    complete specific tasks within deadlines.
    """,

    expected_output="""
    The main types of operating systems include Batch OS,
    Multiprogramming OS, Multitasking OS, Time Sharing OS,
    and Real Time OS.
    """,

    retrieval_context=[
        """
        An Operating System acts as an interface between the user
        and computer hardware. It manages processes, CPU resources,
        files, and other system resources.

        Types of Operating Systems include:

        1. Batch OS - Executes groups of similar jobs sequentially.

        2. Multiprogramming OS - Keeps multiple jobs in memory and
        switches CPU execution when necessary.

        3. Multitasking OS - Performs rapid switching between jobs.

        4. Time Sharing OS - Allows interactive use while sharing
        CPU time between tasks.

        5. Real Time OS - Designed to complete dedicated tasks
        within specified deadlines.
        """
    ],
)


def run_evaluation():
    """Run RAG evaluation and display metric scores."""

    metrics = [
        ("Contextual Precision", contextual_precision),
        ("Contextual Recall", contextual_recall),
        ("Contextual Relevancy", contextual_relevancy),
        ("Answer Relevancy", answer_relevancy),
        ("Faithfulness", faithfulness),
    ]

    print("\nRAG Evaluation Results")
    print("=" * 40)

    for metric_name, metric in metrics:
        metric.measure(test_case)

        print(f"\n{metric_name}")
        print(f"Score: {metric.score}")
        print(f"Reason: {metric.reason}")


if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY was not found. "
            "Add it to your .env file before running evaluation."
        )

    run_evaluation()
