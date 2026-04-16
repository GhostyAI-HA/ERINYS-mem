"""Preference and implicit-connection extraction from conversation text.

Extracts user preferences, habits, opinions, and implicit interests
from raw conversation text using regex patterns, producing synthetic
text fragments for index augmentation.
"""

from __future__ import annotations

import re
from typing import Sequence


PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bI (?:usually |always |generally )?prefer (.{5,80}?)(?:\.|,|$)", re.I),
     "User preference: {}"),
    (re.compile(r"\bI (?:really )?(?:like|love|enjoy) (.{5,80}?)(?:\.|,|$)", re.I),
     "User enjoys: {}"),
    (re.compile(r"\bI (?:don'?t|do not|never) (?:like|want|care for) (.{5,80}?)(?:\.|,|$)", re.I),
     "User dislikes: {}"),
    (re.compile(r"\bmy favorite (?:\w+ )?(?:is|are) (.{5,80}?)(?:\.|,|$)", re.I),
     "User favorite: {}"),
    (re.compile(r"\bI always (.{5,80}?)(?:\.|,|$)", re.I),
     "User habit: always {}"),
    (re.compile(r"\bI tend to (.{5,80}?)(?:\.|,|$)", re.I),
     "User tendency: {}"),
    (re.compile(r"\bI (?:typically|normally|regularly) (.{5,80}?)(?:\.|,|$)", re.I),
     "User habit: {}"),
    (re.compile(r"\bI (?:use|rely on|work with) (.{5,60}?)(?:\.|,|$)", re.I),
     "User uses: {}"),
    (re.compile(r"\bI (?:recommend|suggest) (.{5,80}?)(?:\.|,|$)", re.I),
     "User recommends: {}"),
    (re.compile(r"\bI (?:think|believe|feel) (?:that )?(.{5,80}?)(?:\.|,|$)", re.I),
     "User opinion: {}"),
    (re.compile(r"\b(?:In my (?:opinion|experience|view)),? (.{5,80}?)(?:\.|,|$)", re.I),
     "User opinion: {}"),
    (re.compile(r"\bI (?:still )?remember (.{5,80}?)(?:\.|,|$)", re.I),
     "User memory: {}"),
    (re.compile(r"\bI used to (.{5,80}?)(?:\.|,|$)", re.I),
     "User past habit: {}"),
    (re.compile(r"\b(?:when I was|growing up|back in) (.{5,80}?)(?:\.|,|$)", re.I),
     "User background: {}"),
    (re.compile(r"\bI (?:studied|majored in|graduated from) (.{5,80}?)(?:\.|,|$)", re.I),
     "User education: {}"),
    (re.compile(r"\bI (?:work|worked) (?:at|for|in|as) (.{5,80}?)(?:\.|,|$)", re.I),
     "User career: {}"),
    (re.compile(r"\bI(?:'m| am) (?:a |an )?(.{5,60}?)(?:\.|,|$)", re.I),
     "User identity: {}"),
    (re.compile(r"\bI (?:can'?t stand|hate|detest|loathe) (.{5,60}?)(?:\.|,|$)", re.I),
     "User strongly dislikes: {}"),
    (re.compile(r"\bI(?:'ve| have) (?:always |never )?(?:been|wanted) (.{5,80}?)(?:\.|,|$)", re.I),
     "User aspiration: {}"),
]

IMPLICIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bI(?:'m| am) (?:interested|curious) (?:in|about) (.{5,80}?)(?:\.|,|$)", re.I),
     "User interest: {}"),
    (re.compile(r"\bI(?:'ve| have) been (?:thinking|wondering|looking into) (.{5,60}?)(?:\.|,|$)", re.I),
     "User exploring: {}"),
    (re.compile(r"\bI (?:need|want) (?:to |a |some )?(.{5,80}?)(?:\.|,|$)", re.I),
     "User need: {}"),
    (re.compile(r"\bI(?:'m| am) (?:looking|searching|trying) (?:for|to) (.{5,80}?)(?:\.|,|$)", re.I),
     "User seeking: {}"),
    (re.compile(r"\bI(?:'m| am) (?:struggling|having trouble) with (.{5,60}?)(?:\.|,|$)", re.I),
     "User challenge: {}"),
    (re.compile(r"\bI(?:'m| am) (?:good|experienced|skilled) (?:at|in|with) (.{5,60}?)(?:\.|,|$)", re.I),
     "User skill: {}"),
    (re.compile(r"\bI (?:know how to|can) (.{5,60}?)(?:\.|,|$)", re.I),
     "User capability: {}"),
    (re.compile(r"\bI(?:'m| am) (?:planning|hoping|considering) (?:to |on )?(.{5,80}?)(?:\.|,|$)", re.I),
     "User plan: {}"),
    (re.compile(r"\bI (?:recently|just) (?:started|began|joined|signed up for) (.{5,60}?)(?:\.|,|$)", re.I),
     "User recent activity: {}"),
]


def _apply_patterns(
    text: str,
    patterns: Sequence[tuple[re.Pattern[str], str]],
) -> list[str]:
    """Apply regex patterns to text and return formatted synthetic fragments."""
    results: list[str] = []
    seen: set[str] = set()
    for pattern, template in patterns:
        for m in pattern.finditer(text):
            captured = m.group(1).strip().rstrip(".,;:!?")
            if len(captured) < 4:
                continue
            key = captured.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(template.format(captured))
    return results


def extract_preferences(text: str) -> list[str]:
    """Extract preference/habit/opinion expressions from text.

    Returns a list of synthetic text fragments like:
      "User preference: PostgreSQL over MySQL"
      "User enjoys: hiking on weekends"
    """
    return _apply_patterns(text, PREFERENCE_PATTERNS)


def extract_implicit_connections(text: str) -> list[str]:
    """Extract implicit interest/need/skill expressions from text.

    Returns a list of synthetic text fragments like:
      "User interest: machine learning"
      "User seeking: a new job in data science"
    """
    return _apply_patterns(text, IMPLICIT_PATTERNS)


def extract_all(text: str) -> list[str]:
    """Extract both preferences and implicit connections."""
    return extract_preferences(text) + extract_implicit_connections(text)
