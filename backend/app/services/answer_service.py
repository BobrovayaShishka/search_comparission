import asyncio
import logging

from app.clients.ollama_client import OllamaClient
from app.config import Settings
from app.models.schemas import SearchHit

logger = logging.getLogger(__name__)

_MAX_HITS_IN_PROMPT = 8

_PROMPT_TEMPLATE = """Ты — консультант интернет-магазина. Пользователь спросил: "{query}"

Найденные товары (по релевантности):
{items}

Кратко (2-4 предложения) человеческим языком ответь пользователю на русском:
какие товары ему подходят и почему, опираясь ТОЛЬКО на список выше.
Не придумывай товары и характеристики, которых нет в списке.
Не используй markdown, пиши обычным текстом."""


def _format_items(hits: list[SearchHit]) -> str:
    lines = []
    for i, hit in enumerate(hits[:_MAX_HITS_IN_PROMPT], start=1):
        p = hit.payload
        price = f", цена {p.price:.0f}" if p.price is not None else ""
        desc = f" — {p.description}" if p.description else ""
        lines.append(f"{i}. {p.name} (категория: {p.category or 'не указана'}{price}){desc}")
    return "\n".join(lines)


class AnswerService:
    """
    Generates a short natural-language answer summarizing search results,
    using a lightweight local LLM (Ollama). Purely additive: the raw
    Qdrant hits are always returned regardless of whether this succeeds.

    Callers control this via a request flag (generate_answer). If the flag
    is false, this service is never invoked. If Ollama is disabled, down,
    or times out, the answer field is simply left empty and search still
    returns normally.
    """

    def __init__(self, ollama: OllamaClient, settings: Settings) -> None:
        self._ollama = ollama
        self._settings = settings

    async def build_answer(self, query: str, hits: list[SearchHit]) -> str | None:
        if not self._settings.ollama_enabled:
            return None
        if not hits:
            return "По вашему запросу подходящих товаров не найдено."

        prompt = _PROMPT_TEMPLATE.format(query=query, items=_format_items(hits))

        try:
            return await asyncio.wait_for(
                self._ollama.generate(prompt),
                timeout=self._settings.ollama_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Ollama answer generation timed out")
            return None
        except Exception:
            logger.warning("Ollama answer generation failed", exc_info=True)
            return None
