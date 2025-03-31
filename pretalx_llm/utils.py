import logging
from os import environ

from django.apps import apps
from django.conf import settings
from ollama import Client
from openai import OpenAI

logger = logging.getLogger(__name__)


def get_provider():
    """
    Call this method to get an LlmProvider from the app.
    """
    return apps.get_app_config("pretalx_llm").provider


class LlmProvider:
    """
    This provide access to LLMs.

    An instance of this class will generated in the ready method of the app, which you can get by calling get_provider()
    """

    def __init__(self):
        """
        Generate a new instance of the provider from environment variables or the configuration file.

        A provider can be configured with a string in the format:

            (ollama|openai),name,<url>[,token]

        When the first token is openai, then you can specify the access token by appending a command an an access token to the URL. Ollama doesn't support access token, so the string should be just ollama, followed by a comma and then the URL. The name can be anything human readable, such as for example main, localhost, aws or something else.

        These providers can either be placed in environment variables named LLM_PROVIDER_X, or in a config file in the pretalx_llm section, where the key is named llm_provider_X where X can be anything. When a name is present in the environment variables and in the config file, then the settings in the environment variable takes precedence over the settings in the config file.
        """
        environment_keys = environ.keys()
        env_keys = filter(lambda x: x.startswith("LLM_PROVIDER_"), environment_keys)
        env_values = [self._parse_line(environ.get(x)) for x in env_keys]
        env_map = {name: value for name, value in env_values}

        pretalx_config = settings.PLUGIN_SETTINGS.get("pretalx_llm", {})
        config_keys = filter(
            lambda x: x.startswith("llm_provider_"), pretalx_config.keys()
        )
        config_values = [self._parse_line(pretalx_config.get(x)) for x in config_keys]
        config_map = {name: value for name, value in config_values}

        config_map.update(env_map)

        self.providers = config_map

    def _parse_line(self, line):
        provider_type, name, rest = line.split(",", 2)
        if provider_type == "ollama":
            url = rest
            return (name, OllamaClient(url))
        elif provider_type == "openai":
            url, token = rest.split(",", 1)
            return (name, OpenAiClient(url, token))

    def get_models(self):
        """
        Return a list of (provider,model) of all available models in all configured providers.
        """
        return [
            (provider, model)
            for provider, client in self.providers.items()
            for model in client.get_models()
        ]

    def get_embedding(self, provider, model, input):
        """
        Generate an embedding of a text using a specific provider and model. The input is the text to embed, the return value will be a list of floats (the embedding vector).
        """
        return self.providers[provider].get_embedding(model, input)

    def get_query_embedding(self, provider, model, query):
        """
        Generate an embedding of a query using a specific provider and model. The input is the query to embed, the return value will be a list of floats (the embedding vector).
        """
        return self.providers[provider].get_query_embedding(model, query)


class OllamaClient:
    """
    Implement an LLM provider that accesses LLMs using the Ollama API
    """

    def __init__(self, url):
        self.client = Client(host=url)

    def get_models(self):
        models = [x["model"] for x in list(self.client.list())[0][1]]
        return models

    def get_embedding(self, model, input):
        return self.client.embed(model, input).embeddings[0]

    def get_query_embedding(self, model, input):
        res = self.client.embed(model, "query: {}".format(input)).embeddings[0]
        logger.debug("Res is {}".format(res))
        return res


class OpenAiClient:
    """
    Implement an LLM provider that accesses LLMs using the OpenAI API.

    There are many providers that support this API besides OpenAI itself.
    """

    def __init__(self, url, token):
        self.client = OpenAI(
            base_url=url,
            api_key=token,
        )

    def get_models(self):
        res = self.client.models.list()
        return [x.id for x in res]

    def get_embedding(self, model, input):
        res = self.client.embeddings.create(
            model=model,
            input=[input],
        )
        return res.data[0].embedding

    def get_query_embedding(self, model, query):
        return self.get_embedding(model, "query: {}".format(query))
