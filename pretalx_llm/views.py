import logging
import statistics
from collections import defaultdict

from django.core.exceptions import SuspiciousOperation
from django.db.models import (
    Count,
    Exists,
    F,
    FilteredRelation,
    JSONField,
    OuterRef,
    Q,
    Subquery,
)
from django.db.models.functions import Coalesce
from django.db.utils import IntegrityError
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.template import loader
from django.utils.functional import cached_property
from django.views import View
from django.views.generic import TemplateView
from django_context_decorator import context
from pretalx.common.views.mixins import PermissionRequired
from pretalx.orga.views.submission import ReviewerSubmissionFilter
from pretalx.submission.forms.submission import SubmissionFilterForm
from pretalx.submission.models import Submission
from pretalx.submission.models.review import Review

from . import cache_utils
from .comparison import SubmissionComparison
from .exceptions import LlmModelException
from .forms import (
    ComparisonSettingsForm,
    LimitForm,
    ModelForm,
    PreferencesForm,
    ReviewersSearchForm,
    ReviewStatusForm,
    SearchForm,
    SortForm,
)
from .models import (
    LlmEmbedding,
    LlmEventModels,
    LlmModels,
    LlmUserPreference,
    LlmUserPreferenceEmbedding,
)
from .redislock import PretalxOptionalLock
from .software import SoftwareFilterer
from .tasks import embed_preference, embed_query
from .utils import get_provider

# from silk.profiling.profiler import silk_profile


logger = logging.getLogger(__name__)


class LlmBase(PermissionRequired):

    permission_required = "orga.change_submissions"

    @cached_property
    def models(self):
        return list(
            LlmEventModels.objects.filter(
                event=self.request.event, name__active=True
            ).select_related("name")
        )

    def get_object(self):
        return self.request.event


class LlmPreferences(TemplateView, LlmBase):
    """
    View for the preferences of a user for an event.
    """

    template_name = "pretalx_llm/preferences.html"

    @context
    @cached_property
    def preferences_form(self):
        if self.request.method == "POST":
            return PreferencesForm(
                data=self.request.POST,
            )
        else:
            current_preference = LlmUserPreference.objects.filter(
                user=self.request.user,
                event=self.request.event,
            ).first()
            text = getattr(current_preference, "preference", "")
            return PreferencesForm(initial={"preferenceText": text})

    def post(self, request, event):
        preferences_form = self.preferences_form
        if not preferences_form.is_valid():
            raise SuspiciousOperation()
        preference_text = preferences_form.cleaned_data.get("preferenceText")
        if preference_text == "":
            LlmUserPreference.objects.filter(
                user=self.request.user,
                event=self.request.event,
            ).delete()
        else:
            (pref, created) = LlmUserPreference.objects.update_or_create(
                user=self.request.user,
                event=self.request.event,
                defaults={"preference": preference_text},
            )
            logger.debug("Created is {}".format(created))
            logger.debug("Models are {}".format(self.models))
            for model in self.models:
                logger.debug("Current model is {}".format(model))
                try:
                    embed = LlmUserPreferenceEmbedding.objects.create(
                        preference=preference_text,
                        event_model=model,
                        user_preference=pref,
                    )
                    embed.save()
                    res = embed_preference.apply_async((embed.id,), ignore_result=True)
                    embed.task_id = res.task_id
                    embed.save(update_fields=["task_id"])
                except IntegrityError:
                    # That's fine, then there is already such an embedding
                    pass

        return self.render_to_response(self.get_context_data(event=self.request.event))

    def get_context_data(self, event, **kwargs):
        context = super().get_context_data(**kwargs)

        query = (
            LlmUserPreference.objects.filter(
                event=self.request.event,
            )
            .exclude(
                user=self.request.user,
            )
            .select_related("user")
        )

        context["other_preferences"] = [x for x in query]
        return context


class LlmTemplateView(TemplateView):
    """
    Generic class for Pretalx LLM views that return just an error when no models are available.
    """

    def get(self, request, *args, **kwargs):
        try:
            return super().get(request, *args, **kwargs)
        except LlmModelException:
            return render(
                self.request,
                "pretalx_llm/model_problem.html",
                {},
            )


class LLmGlobalSettingsView(PermissionRequired, TemplateView):
    """
    Server admin view, used to import, enable and disable models.
    """

    permission_required = "person.is_administrator"
    template_name = "pretalx_llm/globalsettings.html"

    def post(self, request):
        action = request.POST["action"]
        provider = get_provider()
        models_on_server = provider.get_models()
        if action == "import":
            r_provider = request.POST["provider"]
            r_name = request.POST["name"]
            r_comment = request.POST["comment"]
            if (r_provider, r_name) not in models_on_server:
                return HttpResponse(status=500)
            local = LlmModels(
                provider=r_provider,
                name=r_name,
                comment=r_comment,
                active=False,
            )
            local.save()
        if action == "start":
            r_provider = request.POST["provider"]
            r_name = request.POST["name"]
            local = LlmModels.objects.get(provider=r_provider, name=r_name)
            if local.active:
                return HttpResponse(status=500)
            local.active = True
            local.save()
        if action == "stop":
            r_provider = request.POST["provider"]
            r_name = request.POST["name"]
            local = LlmModels.objects.get(provider=r_provider, name=r_name)
            if not local.active:
                return HttpResponse(status=500)
            local.active = False
            local.save()
        if action == "delete":
            r_provider = request.POST["provider"]
            r_name = request.POST["name"]
            local = LlmModels.objects.get(provider=r_provider, name=r_name)
            local.delete()

        return self.renderModels(models_on_server)

    def get(self, request):
        provider = get_provider()
        models_on_server = provider.get_models()
        return self.renderModels(models_on_server)

    def renderModels(self, models_on_server):
        modelmap = {
            (model.provider, model.name): model
            for model in LlmModels.objects.annotate(total=Count("llmeventmodels"))
        }
        result = [
            {
                "provider": provider,
                "name": name,
                "imported": (provider, name) in modelmap,
                "count": (
                    modelmap[(provider, name)].total
                    if (provider, name) in modelmap
                    else 0
                ),
                "active": (
                    modelmap[(provider, name)].active
                    if (provider, name) in modelmap
                    else False
                ),
                "comment": getattr(
                    modelmap.get((provider, name)),
                    "comment",
                    "{} - {}".format(provider, name),
                ),
            }
            for provider, name in models_on_server
        ]

        return self.render_to_response({"models": result})


class ReviewFilterHelper(ReviewerSubmissionFilter):
    """
    Helper class to reuse the ReviewerSubmissionFilter class from Pretalx.
    """

    def __init__(self, request):
        self.request = request


class LlmSimilaritiesBase(LlmBase):
    """
    Base class for almost every regular view in Pretalx LLM.
    """

    PERM_VIEW_SPEAKERS = "orga.view_speakers"
    permission_required = "orga.view_orga_area"

    usable_states = None

    @context
    @cached_property
    def can_see_all_reviews(self):
        return self.request.user.has_perm("orga.view_all_reviews", self.request.event)

    @cached_property
    def independent_categories(self):
        return self.request.event.score_categories.all().filter(
            is_independent=True, active=True
        )

    @context
    @cached_property
    def submissions_reviewed(self):
        return Review.objects.filter(
            user=self.request.user, submission__event=self.request.event
        ).values_list("submission_id", flat=True)

    def get_filter_form(self, **kwargs):
        return SubmissionFilterForm(
            data=self.request.GET,
            event=self.request.event,
            usable_states=self.usable_states,
            limit_tracks=self.limit_tracks,
            search_fields=self.get_default_filters(),
            **kwargs
        )

    @context
    @cached_property
    def filter_form(self):
        return self.get_filter_form()

    @cached_property
    def aggregate_method(self):
        return self.request.event.review_settings["aggregate_method"]

    @context
    @cached_property
    def show_submission_types(self):
        return self.request.event.submission_types.all().count() > 1

    @context
    @cached_property
    def limit_form(self):
        return LimitForm(data=self.request.GET)

    @context
    @cached_property
    def review_status_form(self):
        return ReviewStatusForm(self.request.user, data=self.request.GET)

    @context
    @cached_property
    def sort_form(self):
        return SortForm(data=self.request.GET)

    @cached_property
    def review_filter_helper(self):
        return ReviewFilterHelper(self.request)

    @cached_property
    def limit_tracks(self):
        return self.review_filter_helper.limit_tracks

    @context
    @cached_property
    def model_form(self):
        return ModelForm(
            data=self.request.GET,
            models=self.models,
            # Check whether there are also other conditions under which
            # a user can see other reviews, such as the review phase being over
            can_see_other_reviews=self.can_see_all_reviews,
        )

    def get_default_filters(self, *args, **kwargs):
        default_filters = {"code__icontains", "title__icontains"}
        if self.request.user.has_perm(self.PERM_VIEW_SPEAKERS, self.request.event):
            default_filters.add("speakers__name__icontains")
        return default_filters

    def get_base_queryset(self, for_review=False):
        rfh = self.review_filter_helper
        queryset = (
            self.request.event.submissions.all()
            .select_related("submission_type", "event", "track")
            .prefetch_related("speakers")
        )
        if "is_reviewer" in rfh.user_permissions or for_review:
            assigned = self.request.user.assigned_reviews.filter(
                event=self.request.event, pk=OuterRef("pk")
            )
            queryset = queryset.annotate(is_assigned=Exists(Subquery(assigned)))
        if rfh.user_permissions == {"is_reviewer"}:
            queryset = rfh.limit_for_reviewers(queryset)
        return queryset

    def compute_review(self, submission):
        if self.can_see_all_reviews:
            self._compute_review_all(submission)
        else:
            self._compute_review_individual(submission)

    def _compute_review_all(self, submission):
        submission.current_score = (
            submission.median_score
            if self.aggregate_method == "median"
            else submission.mean_score
        )
        if self.independent_categories:  # Assemble medians/means on the fly. Yay.
            independent_ids = [cat.pk for cat in self.independent_categories]
            mapping = defaultdict(list)
            for review in submission.reviews.all():
                for score in review.scores.all():
                    if score.category_id in independent_ids:
                        mapping[score.category_id].append(score.value)
            mapping = {
                key: round(statistics.fmean(value), 1) for key, value in mapping.items()
            }
            result = []
            for category in self.independent_categories:
                result.append(mapping.get(category.pk))
            submission.independent_scores = result

    def _compute_review_individual(self, submission):
        reviews = [
            review
            for review in submission.reviews.all()
            if review.user == self.request.user
        ]
        submission.current_score = None
        if reviews:
            review = reviews[0]
            submission.current_score = review.score
            if self.independent_categories:
                mapping = {
                    score.category_id: score.value for score in review.scores.all()
                }
                result = []
                for category in self.independent_categories:
                    result.append(mapping.get(category.pk))
                submission.independent_scores = result
        elif self.independent_categories:
            submission.independent_scores = [
                None for _ in range(len(self.independent_categories))
            ]

    def annotate_embeddings(self, queryset, model):
        # Subquery to get the embeddings with matching title, abstract and description
        matching_embeddings = LlmEmbedding.objects.filter(
            submission=OuterRef("pk"),
            abstract=OuterRef("abstract"),
            description=OuterRef("description"),
            title=OuterRef("title"),
            event_model=model,
        ).values("embedding")[:1]

        # Subquery that gets the most recent one that is outdated
        recent_embeddings = (
            LlmEmbedding.objects.filter(
                submission=OuterRef("pk"),
                event_model=model,
            )
            .order_by("-created")
            .values("embedding")[:1]
        )

        return queryset.annotate(
            # The embeddings
            embeddings_data=Subquery(
                matching_embeddings, output_field=JSONField()
            )  # Try to match by title, abstract and description
        ).annotate(
            embeddings_data=Coalesce(
                "embeddings_data", Subquery(recent_embeddings, output_field=JSONField())
            )  # Fallback to recent if no match
        )

    def add_reviews(self, model_form, queryset):
        if model_form.show_review_scores():
            user_reviews = Review.objects.filter(
                user=self.request.user, submission_id=OuterRef("pk")
            ).values("score")
            queryset = queryset.annotate(
                # Review of current user
                user_score=Subquery(user_reviews),
            ).prefetch_related(
                "reviews",
                "reviews__user",
                "reviews__scores",
            )
        if model_form.show_review_numbers():
            queryset = queryset.annotate(
                # Review count per submission
                review_count=Count("reviews", distinct=True),
                review_nonnull_count=Count(
                    "reviews", distinct=True, filter=Q(reviews__score__isnull=False)
                ),
            )
        return queryset


class LlmSimilarities(LlmSimilaritiesBase):

    @context
    @cached_property
    def additional_form(self):
        return ComparisonSettingsForm(
            data=self.request.GET,
        )


class LlmReviewerSuggestionsView(LlmSimilaritiesBase, TemplateView):
    """
    _Suggest reviewers for submissions based on the preferences of the reviewers._
    """

    template_name = "pretalx_llm/reviewers.html"

    @context
    @cached_property
    def reviewers_search_form(self):
        return ReviewersSearchForm(
            data=self.request.GET,
        )

    def get_reviewer_queryset(self):
        return LlmUserPreference.objects.filter(
            event=self.request.event,
        ).select_related("user")

    def annotate_reviewer_embeddings(self, queryset, model):
        # Subquery to get the embeddings with matching title and description
        matching_embeddings = LlmUserPreferenceEmbedding.objects.filter(
            user_preference=OuterRef("pk"),
            preference=OuterRef("preference"),
            event_model=model,
        ).values("embedding")[:1]

        # Subquery that gets the most recent one that is outdated
        recent_embeddings = (
            LlmUserPreferenceEmbedding.objects.filter(
                user_preference=OuterRef("pk"),
                event_model=model,
            )
            .order_by("-created")
            .values("embedding")[:1]
        )

        return queryset.annotate(
            # The embeddings
            embeddings_data=Subquery(
                matching_embeddings, output_field=JSONField()
            )  # Try to match by the exact preference
        ).annotate(
            embeddings_data=Coalesce(
                "embeddings_data", Subquery(recent_embeddings, output_field=JSONField())
            )  # Fallback to recent if no match
        )

    def annotate_reviewers(self, queryset):
        queryset = queryset.prefetch_related(
            "reviews",
            "reviews__user",
        )
        return queryset

    def get_context_data(self, event, **kwargs):
        context = super().get_context_data(**kwargs)

        filter_form = self.filter_form
        model_form = self.model_form
        limit_form = self.limit_form
        review_status_form = self.review_status_form
        sort_form = self.sort_form
        reviewers_search_form = self.reviewers_search_form

        if not (
            filter_form.is_valid()
            and model_form.is_valid()
            and limit_form.is_valid()
            and review_status_form.is_valid()
            and sort_form.is_valid()
            and reviewers_search_form.is_valid()
        ):
            raise SuspiciousOperation()

        model = model_form.getModel()

        queryset = self.get_base_queryset()

        # We need the reviewers so that we can check whether a suggested reviewer already reviewed this particular submission
        queryset = self.annotate_reviewers(queryset)

        queryset = self.annotate_embeddings(queryset, model)

        queryset = self.add_reviews(model_form, queryset)
        queryset = self.review_status_form.annotate(queryset)

        queryset = review_status_form.filter_queryset(queryset)
        submission_objects = self.filter_form.filter_queryset(queryset)

        submission_objects = sort_form.order_queryset(submission_objects)
        submission_list = list(
            filter(
                lambda x: x.embeddings_data is not None, submission_objects.distinct()
            )
        )[: limit_form.get_limit()]

        if model_form.show_review_scores():
            for submission in submission_list:
                self.compute_review(submission)

        # Add the reviewers so that the template can say whether this submission was already reviewed by a suggested reviewer
        for submission in submission_list:
            submission.reviewers = [x.user for x in submission.reviews.all()]

        # Find the reviewers now
        reviewer_queryset = self.get_reviewer_queryset()
        reviewer_queryset = self.annotate_reviewer_embeddings(reviewer_queryset, model)
        reviewers = list(
            filter(
                lambda x: x.embeddings_data is not None, reviewer_queryset.distinct()
            )
        )

        sc = SubmissionComparison(submission_list)
        sc.rank_reviewers(reviewers, reviewers_search_form.getN())

        # Return the result
        context["submissions"] = submission_list
        context["show_review_numbers"] = model_form.show_review_numbers()
        context["show_review_scores"] = model_form.show_review_scores()
        return context


class LlmSimilaritiesGraphGeneral(LlmSimilarities):
    """
    Generic class for all graph related views.
    """

    @context
    @cached_property
    def highlight_form(self):
        return self.get_filter_form(prefix="highlight")


class LlmSimilaritiesGraphJson(LlmSimilaritiesGraphGeneral, View):
    """
    View for generating a JSON response that can be used to populate a 2d plot.
    """

    @cached_property
    def can_review(self):
        if self.request.user.has_perm("orga.perform_reviews", self.request.event):
            return True
        return False

    def get_submission_url(self, submission):
        if self.can_review:
            return submission.orga_urls.reviews
        else:
            return submission.orga_urls.base

    @cached_property
    def get_template(self):
        return loader.get_template("pretalx_llm/submission_line.html")

    def render_submission_title(self, submission, model_form):
        # Unfortunately the render code is rather slow, it would be great when the template rendering could be made faster.
        context = {
            "submissions_reviewed": self.submissions_reviewed,
            "show_review_numbers": model_form.show_review_numbers(),
            "show_review_scores": model_form.show_review_scores(),
            "can_see_all_reviews": self.can_see_all_reviews,
            "can_review": self.can_review,
            "submission": submission,
        }
        return self.get_template.render(context, self.request)

    def get(self, request, event):
        model_form = self.model_form
        highlight_form = self.highlight_form

        if not (
            self.filter_form.is_valid()
            and model_form.is_valid()
            and highlight_form.is_valid()
        ):
            raise SuspiciousOperation()

        model = model_form.getModel()

        logger.info("Model is {} and data is {}".format(model, model_form.cleaned_data))

        queryset = self.get_base_queryset()
        queryset = self.annotate_embeddings(queryset, model)

        queryset = self.add_reviews(model_form, queryset)

        sf = SoftwareFilterer(
            highlight_form,
            self.request.user,
            can_search_speakers=self.request.user.has_perm(
                self.PERM_VIEW_SPEAKERS, self.request.event
            ),
        )
        queryset = sf.prefetch(queryset)

        submission_objects = self.filter_form.filter_queryset(queryset)

        submission_list = list(
            filter(
                lambda x: x.embeddings_data is not None, submission_objects.distinct()
            )
        )

        if model_form.show_review_scores():
            for submission in submission_list:
                self.compute_review(submission)

        sc = SubmissionComparison(submission_list)
        sc.add_2d_vectors()

        result = {
            "submissions": [
                {
                    "title": self.render_submission_title(submission, model_form),
                    "x": submission.coords[0],
                    "y": submission.coords[1],
                    "highlight": sf.filter(submission),
                    "url": self.get_submission_url(submission),
                }
                for submission in submission_list
            ]
        }
        return JsonResponse(result)


class LlmSearch(LlmSimilarities, LlmTemplateView):
    """
    Class for text based search using LLM embeddings.

    It can be used to describe the kind of submission you are looking for and then find the best matches.
    """

    template_name = "pretalx_llm/search.html"

    @context
    @cached_property
    def search_form(self):
        return SearchForm(data=self.request.GET)

    def get_context_data(self, event, **kwargs):
        context = super().get_context_data(**kwargs)

        model_form = self.model_form
        search_form = self.search_form
        limit_form = self.limit_form
        review_status_form = self.review_status_form

        if not (
            self.filter_form.is_valid()
            and self.search_form.is_valid()
            and model_form.is_valid()
            and limit_form.is_valid()
            and review_status_form.is_valid()
        ):
            raise SuspiciousOperation()

        model = model_form.getModel()
        search_text = search_form.cleaned_data.get("searchText")

        if search_text == "":
            context["withResuls"] = False
            context["message"] = "You need to enter a query"
            return context

        key = "embed_query_{}_{}".format(
            cache_utils.hash_string(search_text),
            cache_utils.hash_string(str(model.name.pk)),
        )
        with PretalxOptionalLock(
            "user_fulltext_{}".format(self.request.user.get_username())
        ):
            with PretalxOptionalLock("search_fulltext_{}".format(key)):
                embedding = cache_utils.maybe_get(key)
                if embedding is None:
                    result = embed_query.delay(
                        search_text, model.name.provider, model.name.name, key
                    )

                queryset = self.get_base_queryset()
                queryset = self.annotate_embeddings(queryset, model)
                queryset = self.add_reviews(model_form, queryset)
                queryset = self.review_status_form.annotate(queryset)
                queryset = self.review_status_form.filter_queryset(queryset)
                submission_objects = self.filter_form.filter_queryset(queryset)

                submission_list = list(
                    filter(
                        lambda x: x.embeddings_data is not None,
                        submission_objects.distinct(),
                    )
                )

                if model_form.show_review_scores():
                    for submission in submission_list:
                        self.compute_review(submission)

                sc = SubmissionComparison(submission_list)

                if embedding is None:
                    embedding = result.get(timeout=60)

        results = sc.rank_with_query(embedding)
        context["withResults"] = True
        context["submissions"] = results
        context["show_review_numbers"] = model_form.show_review_numbers()
        context["show_review_scores"] = model_form.show_review_scores()
        return context


class LlmReviewSuggesions(LlmSimilarities, LlmTemplateView):
    """
    Suggest submissions to review based on the similarity to those you already reviewed.
    """

    template_name = "pretalx_llm/suggestions.html"

    def get_context_data(self, event, **kwargs):
        context = super().get_context_data(**kwargs)
        model_form = self.model_form
        limit_form = self.limit_form

        if not (
            self.filter_form.is_valid()
            and model_form.is_valid()
            and limit_form.is_valid()
        ):
            raise SuspiciousOperation()

        model = model_form.getModel()

        queryset = self.get_base_queryset()
        queryset = self.annotate_embeddings(queryset, model)
        queryset = self.add_reviews(model_form, queryset)
        submission_objects = self.filter_form.filter_queryset(queryset)

        submission_list = list(
            filter(
                lambda x: x.embeddings_data is not None, submission_objects.distinct()
            )
        )

        # Now the reviewed ones
        reviewed = Submission.objects.filter(reviews__user=self.request.user)
        reviewed = self.annotate_embeddings(reviewed, model)
        reviewed = self.add_reviews(model_form, reviewed)
        reviewed_list = list(
            filter(lambda x: x.embeddings_data is not None, reviewed.distinct())
        )

        if model_form.show_review_scores():
            for submission in submission_list:
                self.compute_review(submission)
            for submission in reviewed_list:
                self.compute_review(submission)

        sc = SubmissionComparison(submission_list)

        if len(reviewed_list) == 0:
            context["withReviews"] = False
            return context
        else:
            context["withReviews"] = True
            # Can we do it more efficiently with an sql query?
            # Pobably not since we need to retrieve the reviewed ones anyway
            left = list(
                filter(
                    lambda x: x not in reviewed_list,
                    sc.rank_with_reviewed(reviewed_list),
                )
            )[: limit_form.get_limit()]

            context["submissions"] = left
            context["show_review_numbers"] = model_form.show_review_numbers()
            context["show_review_scores"] = model_form.show_review_scores()
            return context


class LlmSimilaritiesGraph(LlmSimilaritiesGraphGeneral, LlmTemplateView):
    """
    Display submissions on a 2d plane arranged for similarity.

    This view doesn't do a lot, instead it just generates an HTML page that includes a JavaScript that will then retrieve the data in a separate request.
    """

    template_name = "pretalx_llm/similarities_graph.html"

    # @silk_profile(name="Graph")
    def get_context_data(self, event, **kwargs):
        context = super().get_context_data(**kwargs)

        model_form = self.model_form
        highlight_form = self.highlight_form
        additional_form = self.additional_form

        if not (
            additional_form.is_valid()
            and model_form.is_valid()
            and highlight_form.is_valid()
        ):
            raise SuspiciousOperation()

        context["data_url"] = "graph_json?" + self.request.GET.urlencode()
        return context


class LlmSimilaritiesView(LlmSimilarities, LlmTemplateView):
    """
    Show submissions and submissions that are similar to those.
    """

    template_name = "pretalx_llm/similarities.html"

    # @silk_profile(name="Request")
    def get_context_data(self, event, **kwargs):
        context = super().get_context_data(**kwargs)

        model_form = self.model_form
        additional_form = self.additional_form
        limit_form = self.limit_form
        review_status_form = self.review_status_form
        sort_form = self.sort_form

        if not (
            additional_form.is_valid()
            and self.filter_form.is_valid()
            and model_form.is_valid()
            and limit_form.is_valid()
            and review_status_form.is_valid()
            and sort_form.is_valid()
        ):
            raise SuspiciousOperation()

        model = model_form.getModel()

        logger.debug(
            "Model is {} and data is {}".format(model, model_form.cleaned_data)
        )

        queryset = self.get_base_queryset()
        queryset = self.annotate_embeddings(queryset, model)

        queryset = self.add_reviews(model_form, queryset)
        queryset = self.review_status_form.annotate(queryset)

        with_all = additional_form.cleaned_data["withAll"] == "1"
        if not with_all:
            queryset = review_status_form.filter_queryset(queryset)
            submission_objects = self.filter_form.filter_queryset(queryset)

            def sf(_submission):
                return True

        else:
            software_filter = SoftwareFilterer(
                self.filter_form,
                self.request.user,
                review_status_form=review_status_form,
                can_search_speakers=self.request.user.has_perm(
                    self.PERM_VIEW_SPEAKERS, self.request.event
                ),
            )
            submission_objects = software_filter.prefetch(queryset)
            sf = software_filter.filter

        submission_objects = sort_form.order_queryset(submission_objects)
        logger.debug("query is: {}".format(submission_objects.query))
        logger.debug("dict is: {}".format(submission_objects.query.__dict__))
        submission_list = list(
            filter(
                lambda x: x.embeddings_data is not None, submission_objects.distinct()
            )
        )
        logger.debug("We have {} results".format(len(submission_list)))
        if model_form.show_review_scores():
            for submission in submission_list:
                self.compute_review(submission)

        if len(submission_list) == 0:
            left = []
        else:
            sc = SubmissionComparison(submission_list)
            left = sc.compare_submissions(
                int(additional_form.cleaned_data.get("numberOfSubmissions")),
                sf,
                additional_form.cleaned_data.get("withinTrack") == "0",
            )[: limit_form.get_limit()]

        context["submissions"] = left
        context["show_review_numbers"] = model_form.show_review_numbers()
        context["show_review_scores"] = model_form.show_review_scores()
        return context


class LlmSettingsView(PermissionRequired, TemplateView):
    """
    Enable or disable models for an event.
    """

    permission_required = "orga.change_settings"
    template_name = "pretalx_llm/settings.html"

    def __init__(self, *args, **kwargs):
        logger.debug("Init called")
        super().__init__(*args, **kwargs)

    def get_object(self):
        return self.request.event

    def post(self, request, event):
        global_models = LlmModels.objects.annotate(
            related=FilteredRelation(
                "llmeventmodels",
                condition=Q(llmeventmodels__event=self.request.event.id),
            ),
            event=F("related__event"),
        ).filter(Q(active=True))

        settings = [x for x in self.request.POST.keys() if x.startswith("name_")]
        selections = {
            int(self.request.POST[x]): self.request.POST.get("check_" + x[5:], "")
            == "on"
            for x in settings
        }

        logger.debug("Selections: {}".format(selections))
        logger.debug("Global models: {}".format(global_models.query))

        for model in global_models:
            logger.debug("Model: {} {}".format(model, model.event))
            if model.pk not in selections:
                logger.warning("Could not find {} in selections".format(model.pk))
                continue
            if model.event is None and selections[model.pk] is True:
                logger.info("Creating {} for {}".format(model, event))
                LlmEventModels.objects.create(
                    name=model, event=self.request.event
                ).save()
            if model.event is not None and selections[model.pk] is False:
                logger.info("Deleting {} for event {}".format(model, event))
                LlmEventModels.objects.filter(
                    name=model, event=self.request.event
                ).delete()
        logger.debug("Done looping models")
        return self.get(request, event)

    def get(self, request, event):
        global_models = LlmModels.objects.annotate(
            related=FilteredRelation(
                "llmeventmodels",
                condition=Q(llmeventmodels__event=self.request.event.id),
            ),
            event=F("related__event"),
        ).filter(Q(active=True))
        logger.info("q: {}".format(global_models.query))
        res = list(global_models)
        logger.info(["{}: {}".format(x, x.event) for x in res])
        response = {
            "models": [
                {"name": x.pk, "comment": x.comment, "active": x.event is not None}
                for x in res
            ]
        }
        logger.info(response)
        return self.render_to_response(response)
