import logging
from typing import List

from django import forms
from django.db.models import Exists, OuterRef
from pretalx.common.forms.renderers import InlineFormRenderer
from pretalx.submission.models.review import Review

from .exceptions import LlmModelException
from .models import LlmEventModels

logger = logging.getLogger(__name__)


class ReviewStatusForm(forms.Form):
    """
    Form to indicate whether only submissions reviewed by me or not reviewed by me should be returned.
    """

    default_renderer = InlineFormRenderer

    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user

    reviewStatus = forms.ChoiceField(
        label="review status",
        choices=[
            ("0", "Don't filter based on my review"),
            ("1", "Show only submissions I reviewed"),
            ("2", "Show only submissions I have not reviewed"),
        ],
        initial="0",
        required=False,
    )

    def clean_reviewStatus(self):
        data = self.cleaned_data.get("reviewStatus")
        if not data:
            data = "0"
        return data

    def get_review_status(self):
        return int(self.cleaned_data["reviewStatus"])

    def filter_queryset(self, queryset):
        review_status = self.get_review_status()
        if review_status == 1:
            queryset = queryset.filter(hasreviewed=True)
        elif review_status == 2:
            queryset = queryset.filter(hasreviewed=False)
        return queryset

    def annotate(self, queryset):
        if self.get_review_status() == 0:
            return queryset
        has_reviewed = Review.objects.filter(
            user=self.user, submission_id=OuterRef("pk")
        )
        queryset = queryset.annotate(hasreviewed=Exists(has_reviewed))
        return queryset


class LimitForm(forms.Form):
    """
    Limit the number of results that should be returned.
    """

    default_renderer = InlineFormRenderer

    limit = forms.ChoiceField(
        label="limit",
        choices=[
            ("{}".format((x + 1) * 50), "Show only {} results".format((x + 1) * 50))
            for x in range(5)
        ],
        initial="50",
        required=False,
    )

    def clean_limit(self):
        data = self.cleaned_data.get("limit")
        if not data:
            data = "50"
        return data

    def get_limit(self):
        return int(self.cleaned_data["limit"])


class SearchForm(forms.Form):
    """
    Free text search form for llm based searches.
    """

    searchText = forms.CharField(
        widget=forms.Textarea(attrs={"rows": "5"}),
        required=False,
    )

    default_renderer = InlineFormRenderer

    def clean_searchText(self):
        data = self.cleaned_data.get("searchText")
        if not data:
            data = ""
        return data


class SortForm(forms.Form):
    """
    Specify whether the results should be sorted by code or title.
    """

    default_renderer = InlineFormRenderer

    order = forms.ChoiceField(
        label="Order by",
        required=False,
        choices=[
            ("code", "Order by code"),
            ("codedsc", "Order by code (desc)"),
            ("title", "Order by title"),
            ("titledsc", "Order by title (desc)"),
        ],
        initial=("code", "Order by code"),
    )

    def clean_order(self):
        data = self.cleaned_data.get("order")
        if not data:
            data = "code"
        return data

    def order_queryset(self, queryset):
        choice = self.cleaned_data["order"]
        logger.info("Choice for order is {}".format(choice))
        if choice == "titledsc":
            choice = "-title"
        if choice == "codedsc":
            choice = "-code"
        queryset = queryset.order_by(choice)
        return queryset


class ModelForm(forms.Form):
    """
    Select which model should be used for the llm operations and whether reviews should be shown.
    """

    model = forms.ChoiceField(label="Model", required=False)

    showReview = forms.ChoiceField(label="Show review", initial="00", required=False)

    default_renderer = InlineFormRenderer

    def __init__(
        self,
        models: List[LlmEventModels] = None,
        can_see_other_reviews=True,
        *args,
        **kwargs
    ):
        super(ModelForm, self).__init__(*args, **kwargs)
        if len(models) == 0:
            raise LlmModelException()
        if models is None:
            models = []
        self.fields["model"].choices = [
            (value.name.pk, value.name.comment) for value in models
        ]
        self.fields["model"].initial = models[0]
        self.models = {x.name.pk: x for x in models}
        if can_see_other_reviews:
            self.fields["showReview"].choices = [
                ("00", "Don't show reviews or scores"),
                ("01", "Show number or reviews but no scores"),
                ("10", "Show score but not the number of reviews"),
                ("11", "Show number of reviews and scores"),
            ]
        else:
            self.fields["showReview"].choices = [
                ("00", "Don't show reviews or scores"),
                ("01", "Show number or reviews but no scores"),
                ("10", "My score but not the number of reviews"),
                ("11", "Show number of reviews and my scores"),
            ]

    def clean_showReview(self):
        data = self.cleaned_data.get("showReview")
        if not data:
            data = "00"
        return data

    def show_review_numbers(self):
        return self.cleaned_data.get("showReview")[1] == "1"

    def show_review_scores(self):
        return self.cleaned_data.get("showReview")[0] == "1"

    def clean_model(self):
        logger.info("Cleaned data: {}".format(self.cleaned_data))
        data = self.cleaned_data.get("model")
        if not data:
            data = self.fields["model"].choices[0][0]
        logger.info("Data: {}".format(data))
        return data

    def getModel(self):
        return self.models[int(self.cleaned_data["model"])]


class ComparisonSettingsForm(forms.Form):
    """
    Spedify with which submissions submissions should be compared with.
    """

    withinTrack = forms.ChoiceField(
        label="Within track",
        choices=[("1", "Compare with all tracks"), ("0", "Compare with same track")],
        initial="1",
        required=False,
    )

    withAll = forms.ChoiceField(
        label="With filtered",
        choices=[("1", "Compare with all submissions"), ("0", "Compare with filtered")],
        initial="1",
        required=False,
    )

    numberOfSubmissions = forms.ChoiceField(
        label="Number of submissions to show",
        choices=[
            (str(x), "show {} similar submissions".format(x)) for x in range(1, 11)
        ],
        initial="5",
        required=False,
    )

    default_renderer = InlineFormRenderer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "data" in kwargs and isinstance(kwargs["data"], dict):
            data = kwargs["data"]

            for field_name, field in self.fields.items():
                if field_name not in data or data[field_name] == "":
                    logger.info("Have to fix {}".format(field_name))
                    self.initial[field_name] = field.initial
                    self.data = self.data.copy()
                    self.data[field_name] = field.initial

    def clean_numberOfSubmissions(self):
        data = self.cleaned_data.get("numberOfSubmissions")
        logger.info("Data for the choice is: {}".format(data))
        if not data:
            data = "5"
        return data

    def clean_withinTrack(self):
        data = self.cleaned_data.get("withinTrack")
        if not data:
            data = "1"
        return data

    def clean_withAll(self):
        data = self.cleaned_data.get("withAll")
        if not data:
            data = "1"
        return data
