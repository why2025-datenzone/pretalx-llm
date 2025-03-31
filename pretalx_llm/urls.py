from django.urls import path, re_path
from pretalx.event.models.event import SLUG_REGEX

from .views import (
    LLmGlobalSettingsView,
    LlmReviewSuggesions,
    LlmSearch,
    LlmSettingsView,
    LlmSimilaritiesGraph,
    LlmSimilaritiesGraphJson,
    LlmSimilaritiesView,
)

urlpatterns = [
    re_path(
        rf"^orga/event/(?P<event>{SLUG_REGEX})/settings/p/pretalx_llm/$",
        LlmSettingsView.as_view(),
        name="settings",
    ),
    re_path(
        rf"^orga/event/(?P<event>{SLUG_REGEX})/settings/p/pretalx_llm/similarities$",
        LlmSimilaritiesView.as_view(),
        name="similarities",
    ),
    re_path(
        rf"^orga/event/(?P<event>{SLUG_REGEX})/settings/p/pretalx_llm/graph$",
        LlmSimilaritiesGraph.as_view(),
        name="graph",
    ),
    re_path(
        rf"^orga/event/(?P<event>{SLUG_REGEX})/settings/p/pretalx_llm/graph_json$",
        LlmSimilaritiesGraphJson.as_view(),
        name="graph_json",
    ),
    re_path(
        rf"^orga/event/(?P<event>{SLUG_REGEX})/settings/p/pretalx_llm/suggestions",
        LlmReviewSuggesions.as_view(),
        name="suggestions",
    ),
    re_path(
        rf"^orga/event/(?P<event>{SLUG_REGEX})/settings/p/pretalx_llm/search",
        LlmSearch.as_view(),
        name="search",
    ),
    path(
        "orga/admin/llmsettings/",
        LLmGlobalSettingsView.as_view(),
        name="llmglobalsettings",
    ),
]
