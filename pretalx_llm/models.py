from django.db import models
from pretalx.event.models.event import Event
from pretalx.submission.models import Submission


class LlmModels(models.Model):
    """
    A LLM available in Pretalx.

    Comment is used as a display name so that a user can select or enable it.

    Name and provider are mostly internal so that it can be looked up in the configuration.

    When a model should not be used for a while, it's possible to set active to False, so that it is disabled without deleting all the related embeddings.
    """

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["provider", "name"], name="unique_provider_event"
            )
        ]

    provider = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    comment = models.TextField(blank=True, default="")
    active = models.BooleanField()


class LlmEventModels(models.Model):
    """
    Used to indicate whether a model is used for a specific event.
    """

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["name", "event"], name="unique_event_model")
        ]

    name = models.ForeignKey(LlmModels, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)


class LlmEmbedding(models.Model):
    """
    An embedding vector for a submission.

    Since generating these vectors may take a while, multiple vectors are stored for a submission.

    As long as an embedding is available that matches the current title and description of a submission, this embedding should be used. Should the title or description have changed, then it could be wise to use the most recent embedding vector since there is a good chance that the update was a minor one and the old embedding vector is still useable.

    When a new item is created, the task_id and embedding remain NULL. When the task to generate the embedding is created, the task_id here is updated and once the task has been executed, it will update the embedding vector here.
    """

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["event_model", "submission", "title", "description"],
                name="unique_event_submission_model",
            )
        ]

    event_model = models.ForeignKey(LlmEventModels, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(null=True, blank=True)
    embedding = models.JSONField(null=True)
    task_id = models.CharField(max_length=50)


# class LlmPreferenceEmbedding(models.Model):
#     class Meta:
#         constraints = [
#             models.UniqueConstraint(
#                 fields=["event_model", "user", "text"],
#                 name="unique_preference",
#             )
#         ]

#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     event_model = models.ForeignKey(LlmEventModels, on_delete=models.CASCADE)
#     embedding = models.JSONField(null=True)
#     text = models.TextField()
#     task_id = models.CharField(max_length=50,blank=True)
