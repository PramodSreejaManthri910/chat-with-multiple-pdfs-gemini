from pydantic import BaseModel
import google.generativeai as genai
import instructor

from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)


class CustomGeminiFlash(DeepEvalBaseLLM):
    """Custom Gemini model for DeepEval evaluation."""

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash"
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()

        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )

        response = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )

        return response

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"


# Initialize custom Gemini model
custom_llm = CustomGeminiFlash()


# Initialize evaluation metrics
contextual_precision = ContextualPrecisionMetric(model=custom_llm)
contextual_recall = ContextualRecallMetric(model=custom_llm)
contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)
answer_relevancy = AnswerRelevancyMetric(model=custom_llm)
faithfulness = FaithfulnessMetric(model=custom_llm)


# Test case
test_case = LLMTestCase(
    input="Explain Types of Operating Systems",

    actual_output="""
    1. Batch OS – A set of similar jobs are stored in the main memory for
    execution. A job gets assigned to the CPU only when the execution of
    the previous job completes.

    2. Multiprogramming OS – The main memory consists of jobs waiting for
    CPU time. The OS selects one of the processes and assigns it to the CPU.
    Whenever the executing process needs to wait for another operation,
    such as I/O, the OS selects another process from the job queue.

    3. Multitasking OS – Multitasking OS combines the benefits of
    Multiprogramming OS and CPU scheduling to perform quick switches
    between jobs.

    4. Time Sharing OS – Time-sharing systems require interaction with
    the user to instruct the OS to perform various tasks.

    5. Real Time OS – Real-Time OS are usually built for dedicated systems
    to accomplish a specific set of tasks within deadlines.
    """,

    expected_output="""
    Types of Operating Systems include Batch OS, Multiprogramming OS,
    Multitasking OS, Time Sharing OS, and Real Time OS.
    """,

    retrieval_context=[
        """
        An Operating System can be defined as an interface between user and
        hardware. It is responsible for execution of processes, resource
        allocation, CPU management, file management, and other tasks.

        Types of Operating Systems:

        1. Batch OS – A set of similar jobs are stored in main memory for
        execution.

        2. Multiprogramming OS – Main memory consists of jobs waiting for
        CPU time. The OS selects processes and assigns them to the CPU.

        3. Multitasking OS – Combines Multiprogramming OS and CPU scheduling
        to perform quick switches between jobs.

        4. Time Sharing OS – Requires interaction with the user to instruct
        the OS to perform various tasks.

        5. Real Time OS – Built for dedicated systems to accomplish specific
        tasks within deadlines.
        """
    ],
)


def run_evaluation():
    """Evaluate the RAG response using DeepEval metrics."""

    metrics = [
        ("Contextual Precision", contextual_precision),
        ("Contextual Recall", contextual_recall),
        ("Contextual Relevancy", contextual_relevancy),
        ("Answer Relevancy", answer_relevancy),
        ("Faithfulness", faithfulness),
    ]

    for metric_name, metric in metrics:
        metric.measure(test_case)

        print(f"\n{metric_name} Score: {metric.score}")
        print(f"Reason: {metric.reason}")


if __name__ == "__main__":
    run_evaluation()
