import hashlib
import json

from django.db import migrations, models


def _compute_content_hash(title, abstract, description):
    to_hash = json.dumps([title, abstract, description])
    return hashlib.sha256(to_hash.encode("utf-8")).hexdigest()


def generate_content_hash(apps, schema_editor):
    embeddings = apps.get_model("pretalx_llm", "LlmEmbedding")
    for emb in embeddings.objects.iterator():
        contenthash = _compute_content_hash(emb.title, emb.abstract, emb.description)
        emb.contenthash = contenthash
        emb.save()


class Migration(migrations.Migration):

    dependencies = [
        ("pretalx_llm", "0004_delete_old_embeddings"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="LlmEmbedding",
            name="unique_event_submission_model",
        ),
        migrations.AddField(
            model_name="LlmEmbedding",
            name="contenthash",
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.RunPython(
            generate_content_hash, reverse_code=migrations.RunPython.noop
        ),
        migrations.AlterField(
            model_name="LlmEmbedding",
            name="contenthash",
            field=models.CharField(max_length=200, null=False),
        ),
        migrations.AddConstraint(
            model_name="LlmEmbedding",
            constraint=models.UniqueConstraint(
                fields=("event_model", "submission", "contenthash"),
                name="unique_event_submission_model",
            ),
        ),
    ]
