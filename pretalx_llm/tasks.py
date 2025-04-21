import logging

from pretalx.celery_app import app

from . import cache_utils
from .models import LlmEmbedding, LlmUserPreferenceEmbedding
from .utils import get_provider

logger = logging.getLogger(__name__)


@app.task(
    name="pretalx_llm.signals.embed_submission",
    autoretry_for=(Exception,),
    max_retries=10,
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def embed_submission(id):
    """
    Generate the embedding vector for a submission. The actual LlmEmbedding object was already created by the caller and the primary key is passed as an argument.
    """
    logger.debug("Starting async task: {}".format(id))
    try:
        embed = LlmEmbedding.objects.get(id=id)
        if embed.embedding is not None:
            # The embedding vector is already present. Maybe this task was already completed but for some reasons called a second time. Just abort with success here to save resources.
            logger.warning("Embedding already present for id {}".format(id))
            return
        model_provider = get_provider()
        to_input = "Title: {}\n\nAbstract: {}\n\nDescription: {}".format(embed.title, (embed.abstract or ""), (embed.description or ""))
        logger.debug("Input for embedding is: {}".format(to_input))
        result = model_provider.get_embedding(
            embed.event_model.name.provider, embed.event_model.name.name, to_input
        )
        logger.debug("Embedding result for {} is: {}".format(id, result))
        embed.embedding = result
        embed.save(update_fields=["embedding"])
        logger.info("Success for task {}".format(id))
    except Exception as err:
        logger.error("Failed to run task {}: {}".format(id, err))
        raise err


@app.task(
    name="pretalx_llm.signals.embed_query",
    autoretry_for=(Exception,),
    # Short backoff time and not many retries since a user will not wait that long for search results
    max_retries=5,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def embed_query(query, provider, model, key=None):
    """
    Embed a query for a given provider and model. The key can be a key under which this result should be cached.
    """
    logger.debug("Starting async query task: {}".format(query))
    try:
        model_provider = get_provider()
        result = model_provider.get_query_embedding(provider, model, query)
        logger.debug("Embedding result for {} is: {}".format(id, result))
        if key is not None:
            cache_utils.maybe_set(key, result, timeout=3600 * 2)
        return result
    except Exception as err:
        logger.error("Failed to run task {}: {}".format(id, err))
        raise err


@app.task(
    name="pretalx_llm.signals.embed_preference",
    autoretry_for=(Exception,),
    max_retries=10,
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def embed_preference(preference_embedding_id):
    """
    Embed a user preference for a given provider and model.
    """
    logger.debug("Starting async preference task: {}".format(preference_embedding_id))
    try:
        embed = LlmUserPreferenceEmbedding.objects.get(id=preference_embedding_id)
        model_provider = get_provider()
        embedding = model_provider.get_query_embedding(
            embed.event_model.name.provider,
            embed.event_model.name.name,
            embed.preference,
        )
        embed.embedding = embedding
        embed.save(update_fields=["embedding"])

    except Exception as err:
        logger.error("Failed to run task {}: {}".format(id, err))
        raise err
