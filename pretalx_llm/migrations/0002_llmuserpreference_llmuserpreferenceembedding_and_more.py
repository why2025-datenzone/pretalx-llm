# Generated by Django 5.1.6 on 2025-04-06 00:21

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("event", "0036_alter_event_header_image_alter_event_logo"),
        ("pretalx_llm", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="LlmUserPreference",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True, primary_key=True, serialize=False
                    ),
                ),
                ("preference", models.TextField()),
                (
                    "event",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="event.event"
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="LlmUserPreferenceEmbedding",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True, primary_key=True, serialize=False
                    ),
                ),
                ("created", models.DateTimeField(auto_now_add=True, null=True)),
                ("preference", models.TextField()),
                ("embedding", models.JSONField(null=True)),
                ("task_id", models.CharField(max_length=50)),
                (
                    "event_model",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="pretalx_llm.llmeventmodels",
                    ),
                ),
                (
                    "user_preference",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="pretalx_llm.llmuserpreference",
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="llmuserpreference",
            constraint=models.UniqueConstraint(
                fields=("event", "user"), name="unique_event_user_model"
            ),
        ),
        migrations.AddConstraint(
            model_name="llmuserpreferenceembedding",
            constraint=models.UniqueConstraint(
                fields=("event_model", "user_preference", "preference"),
                name="unique_event_user_model_embedding",
            ),
        ),
    ]
