from django.apps import AppConfig
from django.utils.translation import gettext_lazy

from . import __version__
from .utils import LlmProvider


class LlmApp(AppConfig):
    name = "pretalx_llm"
    verbose_name = "Pretalx LLM"

    class PretalxPluginMeta:
        name = gettext_lazy("Pretalx LLM")
        author = "Erik Tews"
        description = gettext_lazy(
            "Pretalx LLM integration for semantic similarity of submissions"
        )
        visible = True
        version = __version__
        category = "FEATURE"

    def ready(self):
        from . import signals  # NOQA
        from . import tasks  # NOQA

        # from . import urls
        self.provider = LlmProvider()


default_app_config = "pretalx_llm.LlmApp"
