Pretalx LLM
==========================

This is a plugin for `pretalx`_.
Pretalx LLM integration for semantic similarity of submissions

Features
--------

In genreal, the plugin offers the following features:

1. Show related submissions for every submission that has been submitted. This can be helpful for reviewers who want to compare submissions with similar ones when reviewing them.

2. Visualize the similarity between submission on a two dimensional plain. This can be helpful to identify clusters of submissions with related content.

3. Show a user submissions that are similar to ones the user has already reviewed.

4. Search for submissions based on a textual description of interesting topics. For example a user can search for **enterprise information security** and the plugin will show submissions that match the topic, even though when the submission doesn't have any of the words in the title or description.

4. Each reviewer can set some review preferences (a short text that describes what kind of submissions they are interested in, such as **enterprise information security**) and based on those preferences, matching reviewers are then suggested for each submission.

Development setup
-----------------

1. Make sure that you have a working `pretalx development setup`_.

2. Clone this repository, eg to ``local/pretalx-llm``.

3. Activate the virtual environment you use for pretalx development.

4. Run ``pip install -e .`` within this directory to register this application with pretalx's plugin registry.

5. Run ``make`` within this directory to compile translations.

6. Restart your local pretalx server. This plugin should show up in the plugin list shown on startup in the console.
   You can now use the plugin from this repository for your events by enabling it in the 'plugins' tab in the settings.

This plugin has CI set up to enforce a few code style rules. To check locally, you need these packages installed::

    pip install flake8 flake8-bugbear isort black

To check your plugin for rule violations, run::

    black --check .
    isort -c .
    flake8 .

You can auto-fix some of these issues by running::

    isort .
    black .

Requirements
------------

1. You should definitely run Pretalx with Celery enabled. For that, you will need a Redis server and configure Pretalx properly.

2. You need some kind of LLM provider. You can self-host everything by using Ollama or another solution that supports the OpenAI API.

You can run both services for development using the following ``docker-compose.yml`` file::

    version: '3.8'
    services:
    cache:
        image: redis:latest
        restart: always
        ports:
        - '127.0.0.1:6379:6379'
        command: redis-server --save 20 1 --loglevel warning 
        volumes: 
        - cache:/data

    ollama:
        image: ollama/ollama:latest
        ports:
        - 127.0.0.1:11434:11434
        volumes:
        - ollama:/root/.ollama
        container_name: ollama
        pull_policy: always
        tty: true
        restart: always
        environment:
        - OLLAMA_KEEP_ALIVE=24h
        - OLLAMA_HOST=0.0.0.0

    volumes:
    ollama:
        driver: local
    cache:
        driver: local

You may want to adopt this file a bit for production usage. Keep in mind that Ollama doesn't have any kind of authentication, everyone who can reach the Ollama server via HTTP is able to use it. That's why it's bound to the loopback address only here.

Configuration
-------------

You need to configure an LLM provider so that Pretalx knows where to find it. The plugin can be configured in two ways:

Configuration file
^^^^^^^^^^^^^^^^^^

You can configure the plugin using the pretalx configuration file::

    [plugin:pretalx_llm]
    llm_provider_1=ollama,main,http://localhost:11434/

This configures a single provider named **main** using the Ollama API that is taking requests at ``http://localhost:11434/``. When you have multiple providers, then add more lines using **llm_provider_X** as a key, and X can be an arbitrary value. It's only important that each provider has a unique name.

Another provider using the OpenAI API could be configured like that::

    llm_provider_2=openai,openaiprovider,http://some-other-host/v1,mytoken

This configures a second provider using the OpenAI protocol name **openaiprovider** that points to the API URL **http://some-other-host/v1**. OpenAI requires an access token, which is set to **mytoken** here. When you use Ollama with the OpenAI API, then you still need to set a token, but it will be ignored by Ollama.

Environment variables
^^^^^^^^^^^^^^^^^^^^^

Alternatively you can set environment variables, for example like that::

    LLM_PROVIDER_1=ollama,main,http://localhost:11434/
    LLM_PROVIDER_2=openai,openaiprovider,http://some-other-host/v1,mytoken

Environment variables overwrite config file settings when they specify an provider with a name that already exists in the config file.

Restart Pretalx
^^^^^^^^^^^^^^^

Don't forget to restart Pretalx, including the Celery runner after you changed the configuration.

Models
------

Pretalx LLM uses models for embeddings only. In general you can pick any model you want. We recommend that you pick a model with a sufficiently large context window so that the title and description of a submission fit in the context window of the model. Otherwise the model should support all the languages that speakers use for their submissions. In general, models with more parameters yield in better results, but they are slower. When you run your LLM provider on a system with a good GPU or AI hardware accelerator, then more powerful models are a good option. However there are also models that run still well on regular CPUs and still produce good results.

In general, **snowflake-arctic-embed2** is a good start. Creating an embedding takes just a few seconds (often less) on a moderately fast Ultrabook from 2018 and it can handle common western languages well.

Setting up the model
^^^^^^^^^^^^^^^^^^^^

When your local Ollama server is up and running, then you can pull a model using the following curl command::

    curl http://localhost:11434/api/pull -d '{
    "model": "snowflake-arctic-embed2"
    }'

It could take seconds to minutes to download the model, depending on your internet connection speed.

Settings
--------

There are global settings and per event settings. In general models need to be globally imported by the administrator of the instance and can then be enabled by the event organizers.

Admin settings
^^^^^^^^^^^^^^

Once you configured your provider and made some models available, you should login as admin and go to the **LLM Global Settings** and import a model there. You should see a list of all available models there.

You can give the model a name and you may also want to include hints for the users. For example you could name your model **snowflake-arctic-embed2 (great for most conferences)**. This name will then be shown to event organizers and reviewers. Once you are satisfied with the name, click on the import icon. Once the model is imported, enable it with the play button.

Per Event settings
^^^^^^^^^^^^^^^^^^

Once an admin has enabled a model, event organizers can enable the plugin in the event settings and then enable some of the models in the event settings.

After a model has been enabled for an event, pretalx will start indexing all submissions for this event. This may take a while.

Once the submissions are indexed, the plugin can be used.

Improvements
------------

There are a few ways how Pretalx LLM could be improved, but that depends on Pretalx upstream:

1. **Support for priorities in Celery.** Right now, Pretalx doesn't support job priorities for Celery with Redis. There are some tasks that have low priority, such as re-indexing a submission, and there are some jobs that have a high priority, such as generating the embedding vector for a query. Right now, indexing of submissions could temporarily render the semantic search feature unavailable.

2. **Vector searches in the database.** Right now, there are extensions for Sqlite and Postgres that support vector searches, such as finding vectors with a low distance to a given target vector. Having support for that could make Pretalx LLM faster. In the current implementation, all the relevant embedding vectors are loaded in the application from the database and then compared and ranked there. That works fine, but when this could be done right in the database then this could potentially be faster and it could reduce the network traffic from the database to the application and it could lower the memory usage of the application itself.

License
-------

Copyright 2025 Erik Tews

Released under the terms of the Apache License 2.0

Please keep in mind that some of the code is actually copy&paste from Pretalx since some of the code of Pretalx was hard to use directly without copy&pasting it into the plugin.

.. _pretalx: https://github.com/pretalx/pretalx
.. _pretalx development setup: https://docs.pretalx.org/en/latest/developer/setup.html
