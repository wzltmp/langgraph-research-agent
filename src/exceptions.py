"""Custom exception types for the research agent.

Using named exceptions instead of bare ``RuntimeError`` / ``ValueError`` lets
callers handle them precisely and makes failures self-documenting in logs.
"""
from __future__ import annotations


class ResearchAgentError(Exception):
    """Base for all agent-specific errors."""


class EmptyLLMResponseError(ResearchAgentError):
    """The LLM returned a message with no usable text content (refusal or empty)."""


class PlanParseError(ResearchAgentError):
    """The planner's reply could not be parsed into a list of sub-queries."""
