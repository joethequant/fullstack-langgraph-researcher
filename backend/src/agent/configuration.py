"""Configuration settings for the research agent."""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="o3-mini",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="o3-mini",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    reasoning_model: str = Field(
        default="o3-mini",
        metadata={
            "description": "The name of the language model to use for the agent's reasoning."
        },
    )

    answer_model: str = Field(
        default="o3-mini",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    openai_query_model: str = Field(
        default="o3-mini",
        metadata={
            "description": "Optional OpenAI model used for query generation.",
        },
    )

    openai_answer_model: str = Field(
        default="o3-mini",
        metadata={
            "description": "Optional OpenAI model used for answer generation.",
        },
    )   

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config with environment variables
        # taking precedence over values supplied via ``configurable``.
        raw_values: dict[str, Any] = {}
        for name in cls.model_fields.keys():
            env_value = os.environ.get(name.upper())
            if env_value is not None:
                raw_values[name] = env_value
            elif configurable.get(name) is not None:
                raw_values[name] = configurable.get(name)

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
