import logging

from django.db.models import F, FilteredRelation, Q
from django.db.utils import IntegrityError
from django.dispatch import receiver
from django.urls import resolve, reverse
from django.utils.translation import gettext_lazy as _
from django_scopes import scope
from pretalx.common.signals import minimum_interval, periodic_task
from pretalx.orga.signals import nav_event, nav_event_settings, nav_global
from pretalx.submission.models import Submission

from .models import LlmEmbedding, LlmUserPreference, LlmUserPreferenceEmbedding
from .tasks import embed_preference, embed_submission

logger = logging.getLogger(__name__)


@receiver(nav_global)
def admin_menu(sender, request, **kwargs):
    """
    Generate the global admin menu entry.
    """
    if not request.user.is_administrator:
        return []

    url = resolve(request.path_info)

    return [
        {
            "label": _("LLM Global Settings"),
            "url": "/orga/admin/llmsettings/",
            "icon": "connectdevelop",
            "active": (
                (url.url_name == "llmglobalsettings")
                and (url.namespaces == ["plugins", "pretalx_llm"])
            ),
        }
    ]


@receiver(nav_event)
def pretalx_llm_settings(sender, request, **kwargs):
    if not request.user.has_perm("orga.view_orga_area", request.event):
        return []
    url = resolve(request.path_info)
    result = [
        {
            "label": "Pretax LLM",
            "icon": "connectdevelop",
            "url": reverse(
                "plugins:pretalx_llm:similarities",
                kwargs={"event": request.event.slug},
            ),
            "active": (
                (url.url_name in ("similarities", "graph", "suggestions", "search"))
                and (url.namespaces == ["plugins", "pretalx_llm"])
            ),
            "children": [
                {
                    "label": "Similarities",
                    "url": reverse(
                        "plugins:pretalx_llm:similarities",
                        kwargs={"event": request.event.slug},
                    ),
                    "active": (
                        (url.url_name == "similarities")
                        and (url.namespaces == ["plugins", "pretalx_llm"])
                    ),
                },
                {
                    "label": "Graph",
                    "url": reverse(
                        "plugins:pretalx_llm:graph",
                        kwargs={"event": request.event.slug},
                    ),
                    "active": (
                        (url.url_name == "graph")
                        and (url.namespaces == ["plugins", "pretalx_llm"])
                    ),
                },
                {
                    "label": "Suggestions",
                    "url": reverse(
                        "plugins:pretalx_llm:suggestions",
                        kwargs={"event": request.event.slug},
                    ),
                    "active": (
                        (url.url_name == "suggestions")
                        and (url.namespaces == ["plugins", "pretalx_llm"])
                    ),
                },
                {
                    "label": "Search",
                    "url": reverse(
                        "plugins:pretalx_llm:search",
                        kwargs={"event": request.event.slug},
                    ),
                    "active": (
                        (url.url_name == "search")
                        and (url.namespaces == ["plugins", "pretalx_llm"])
                    ),
                },
                {
                    "label": "Preferences",
                    "url": reverse(
                        "plugins:pretalx_llm:preferences",
                        kwargs={"event": request.event.slug},
                    ),
                    "active": (
                        (url.url_name == "preferences")
                        and (url.namespaces == ["plugins", "pretalx_llm"])
                    ),
                },
                {
                    "label": "Reviewers",
                    "url": reverse(
                        "plugins:pretalx_llm:reviewers",
                        kwargs={"event": request.event.slug},
                    ),
                    "active": (
                        (url.url_name == "reviewers")
                        and (url.namespaces == ["plugins", "pretalx_llm"])
                    ),
                },
            ],
        }
    ]
    return result


@receiver(nav_event_settings)
def pretalx_llm_settings_settings(sender, request, **kwargs):
    """
    Create the menu entry that links to the per event LLM settings.
    """
    if not request.user.has_perm("orga.change_settings", request.event):
        return []
    return [
        {
            "label": "Pretalx LLM",
            "url": reverse(
                "plugins:pretalx_llm:settings",
                kwargs={"event": request.event.slug},
            ),
            "active": request.resolver_match.url_name == "plugins:pretalx_llm:settings",
        }
    ]


@receiver(signal=periodic_task)
@minimum_interval(minutes_after_success=1, minutes_after_error=1)
def run_llm_preference_reindex(sender, **kwargs):
    try:
        with scope(event=None):
            query = LlmUserPreference.objects.annotate(
                modelname=F("event__llmeventmodels__name"),
                eventmodelid=F("event__llmeventmodels__id"),
                modelstatus=F("event__llmeventmodels__name__active"),
                embedding=FilteredRelation(
                    "llmuserpreferenceembedding",
                    condition=(
                        Q(
                            llmuserpreferenceembedding__event_model=F(
                                "event__llmeventmodels"
                            )
                        )
                        & Q(llmuserpreferenceembedding__preference=F("preference"))
                    ),
                ),
                embid=F("embedding__id"),
            ).filter(modelstatus=True, embid__isnull=True, modelname__isnull=False)
            for missing_preference_embedding in list(query):
                try:
                    embed = LlmUserPreferenceEmbedding.objects.create(
                        preference=missing_preference_embedding.preference,
                        event_model_id=missing_preference_embedding.eventmodelid,
                        user_preference=missing_preference_embedding,
                    )
                    embed.save()
                    res = embed_preference.apply_async((embed.id,), ignore_result=True)
                    embed.task_id = res.task_id
                    embed.save(update_fields=["task_id"])
                except IntegrityError:
                    # That's fine, then there is already such an embedding
                    pass
                except Exception as err:
                    logger.info("Failed to embed user preference: {}".format(err))

    except Exception as e:
        logger.info("Exception: {}".format(e))


@receiver(signal=periodic_task)
@minimum_interval(minutes_after_success=1, minutes_after_error=1)
def run_llm_reindex(sender, **kwargs):
    """
    In a nutshell, this method tries to find all submissions that don't have a matching LlmEmbedding yet for every active event model. It will then create the missing LlmEmbeddings and start a celery task that will generate the embedding vector.
    """
    logger.info("running indexing")
    try:
        with scope(event=None):
            query = Submission.objects.annotate(
                modelname=F("event__llmeventmodels__name"),
                eventmodelid=F("event__llmeventmodels__id"),
                modelstatus=F("event__llmeventmodels__name__active"),
                embedding=FilteredRelation(
                    "llmembedding",
                    condition=(
                        Q(llmembedding__event_model=F("event__llmeventmodels"))
                        & Q(llmembedding__title=F("title"))
                        & (
                            Q(llmembedding__description=F("description"))
                            | (
                                Q(llmembedding__description__isnull=True)
                                & Q(description__isnull=True)
                            )
                        )
                    ),
                ),
                embid=F("embedding__id"),
            ).filter(modelstatus=True, embid__isnull=True, modelname__isnull=False)
        logger.info("Query to get all missing embeddings: {}".format(query.query))
    except Exception as e:
        logger.info("Exception: {}".format(e))
    for missing_embedding in list(query):
        try:
            # Create the LlmEmbedding without task_id and embedding vector
            embed = LlmEmbedding.objects.create(
                submission=missing_embedding,
                title=missing_embedding.title,
                description=missing_embedding.description,
                event_model_id=missing_embedding.eventmodelid,
            )
            embed.save()
        except Exception as e:
            logger.warning(
                "Failed to create embedding for title: {} description: {}: {}".format(
                    missing_embedding.title, missing_embedding.description, e
                )
            )
            continue
        try:
            # Start the task and update the LlmEmbedding with the task_id.
            res = embed_submission.apply_async((embed.id,), ignore_result=True)
            embed.task_id = res.task_id
            embed.save(update_fields=["task_id"])
            # The celery task will then update the LlmEmbedding with the embeddings vector.
        except Exception as err:
            logger.warning("Exception: {}".format(err))
            raise err
