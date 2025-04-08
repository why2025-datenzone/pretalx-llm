import itertools
import logging
from datetime import timedelta

import celery
from celery.result import AsyncResult
from django.db import transaction
from django.db.models import Exists, F, FilteredRelation, OuterRef, Q
from django.db.models.signals import post_save
from django.db.utils import IntegrityError
from django.dispatch import receiver
from django.urls import resolve, reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django_scopes import scope
from pretalx.celery_app import app
from pretalx.common.signals import minimum_interval, periodic_task
from pretalx.orga.signals import nav_event, nav_event_settings, nav_global
from pretalx.submission.models import Submission

from .models import (
    LlmEmbedding,
    LlmEventModels,
    LlmUserPreference,
    LlmUserPreferenceEmbedding,
)
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
@minimum_interval(minutes_after_success=60, minutes_after_error=60)
def remove_failed_tasks(sender, **kwargs):
    """
    _Delete embeddings that have a task id but the corresponding task failed_

    This is a bit more expensive since we need to query Celery for every task that is currently in the queue.
    """
    try:
        logger.debug("Starting to check for failed tasks")
        one_hour_ago = timezone.now() - timedelta(hours=1)

        # Now lets get those that have a task id but the embedding isn't there yet and delete the failed ones
        user_embeddings = LlmUserPreferenceEmbedding.objects.filter(
            task_id__isnull=False,
            created__lt=one_hour_ago,
            embedding__isnull=True,
        )

        embeddings = LlmEmbedding.objects.filter(
            task_id__isnull=False,
            created__lt=one_hour_ago,
            embedding__isnull=True,
        )

        for embedding in itertools.chain(user_embeddings, embeddings):
            result = AsyncResult(embedding.task_id, app=app)
            if result.state in [celery.states.REVOKED, celery.states.FAILURE]:
                logger.warning("A task with id {} failed!".format(embedding.task_id))
                embedding.delete()

        logging.debug("Finished deleting failed tasks")
    except Exception as e:
        logger.error("Failed to check for failed tasks: {}".format(e))
        raise e


@receiver(signal=periodic_task)
@minimum_interval(minutes_after_success=10, minutes_after_error=10)
def remove_embeddings(sender, **kwargs):
    """
    _Delete embeddings that have either no task id and were created more than an hour ago or those that are outdated._
    """

    try:
        logging.debug("Cleaning up embeddings")

        # Get embeddings without task ids that were created more an an hour ago and delete them
        one_hour_ago = timezone.now() - timedelta(hours=1)

        user_deleted, _ = LlmUserPreferenceEmbedding.objects.filter(
            task_id__isnull=True,
            created__lt=one_hour_ago,
        ).delete()

        embedding_deleted, _ = LlmEmbedding.objects.filter(
            task_id__isnull=True,
            created__lt=one_hour_ago,
        ).delete()

        if (user_deleted + embedding_deleted) > 0:
            # In general, tasks should not fail, notify the user
            logger.warning(
                "Deleted {} user preference embeddings and {} submission embeddings without a task id".format(
                    user_deleted, embedding_deleted
                )
            )
        else:
            logger.debug("There were no embeddings without a task id that were deleted")

        # Now get those that have a more recent embedding and they are undated (different preference)
        newer_user_embedding_exists = LlmUserPreferenceEmbedding.objects.filter(
            user_preference=OuterRef("user_preference"),
            event_model=OuterRef("event_model"),
            embedding__isnull=False,
            created__gt=OuterRef("created"),
        )

        outdated_user_embeddings = LlmUserPreferenceEmbedding.objects.filter(
            Exists(newer_user_embedding_exists),
            embedding__isnull=False,
        ).exclude(preference=F("user_preference__preference"))

        logger.debug(
            "Query for outdated ones is: {}".format(outdated_user_embeddings.query)
        )
        outdated_user_embeddings.delete()

        newer_embedding_exists = LlmEmbedding.objects.filter(
            event_model=OuterRef("event_model"),
            submission=OuterRef("submission"),
            embedding__isnull=False,
            created__gt=OuterRef("created"),
        )

        outdated_embeddings = LlmEmbedding.objects.filter(
            Exists(newer_embedding_exists),
            embedding__isnull=False,
        ).exclude(
            Q(title=F("submission__title"))
            & Q(
                Q(description=F("submission__description"))
                | (
                    Q(submission__description__isnull=True)
                    & Q(description__isnull=True)
                )
            )
        )

        logger.debug(
            "Query for SECOND outdated ones is: {}".format(outdated_embeddings.query)
        )
        outdated_embeddings.delete()

    except Exception as e:
        logger.warning("Exception while trying to cleanup embeddings: {}".format(e))


@receiver(signal=periodic_task)
@minimum_interval(minutes_after_success=1, minutes_after_error=1)
def run_llm_preference_reindex(sender, **kwargs):
    """
    _Create embedding vectors for all user preferences that are currently missing or outdated._
    """
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
                    logger.warning("Failed to embed user preference: {}".format(err))

    except Exception as e:
        logger.warning("Exception: {}".format(e))


@receiver(signal=periodic_task)
@minimum_interval(minutes_after_success=1, minutes_after_error=1)
def run_llm_reindex(sender, **kwargs):
    """
    In a nutshell, this method tries to find all submissions that don't have a matching LlmEmbedding yet for every active event model. It will then create the missing LlmEmbeddings and start a celery task that will generate the embedding vector.
    """
    logger.debug("running indexing")
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
        logger.debug("Query to get all missing embeddings: {}".format(query.query))
        for missing_embedding in list(query):
            embed_single_submission(missing_embedding, missing_embedding.eventmodelid)
    except Exception as e:
        logger.warning("Exception: {}".format(e))


def embed_single_submission(submission: Submission, model_id=None):
    """
    _Generate embeddings for a single submission._

    When a model id is given, it will only generate embeddings for this particular model. Otherwise, embeddings for all active models will be generated.

    Args:
        submission (Submission): _The submission to create embeddings for._
        model_id (_int_, optional): _An optional id of an event model_. Defaults to None.
    """
    if model_id is None:
        model_ids = [
            x.id
            for x in LlmEventModels.objects.filter(
                event=submission.event, name__active=True
            )
        ]
    else:
        model_ids = [model_id]
    for current_model_id in model_ids:
        try:
            # Create the LlmEmbedding without task_id and embedding vector
            embed = LlmEmbedding.objects.create(
                submission=submission,
                title=submission.title,
                description=submission.description,
                event_model_id=current_model_id,
            )
            embed.save()
        except IntegrityError:
            # That's fine, there is already such an embedding
            continue
        except Exception as e:
            logger.warning(
                "Failed to create embedding for title: {} description: {}: {}".format(
                    submission.title, submission.description, e
                )
            )
            continue
        try:
            # Start the task and update the LlmEmbedding with the task_id.
            res = embed_submission.apply_async((embed.id,), ignore_result=True)
            embed.task_id = res.task_id
            embed.save(update_fields=["task_id"])
        except Exception as e:
            logger.warning("Could not start task for submission save: {}".format(e))


@receiver(post_save, sender=Submission)
def update_submission_index(instance, **kwargs):
    """_Will be called when a Submission is saved._

    It registers an on_commit hook so that embeddings will be generated once the submission is committed to the database.

    Args:
        instance (_Submission_): _The submission that was saved._
    """
    transaction.on_commit(lambda: embed_single_submission(instance))
