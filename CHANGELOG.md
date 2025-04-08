# Version 0.1.0

* Reviewers can now indicate their preferences for submissions in text form.
* Based on those preferences, the best matching reviewers are then shown next to the submissions.
* Embeddings that are outdated are now periodically removed from the database.
* Should a Celery job fail to execute, those stall embeddings are periodically removed as well.
* When a submission is saved, the generation of a new embedding vector for this submission will be instantly triggered.

# Version 0.0.1

* Initial version