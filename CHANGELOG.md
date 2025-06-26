# Version 0.5.1

* Store the hash of the content of a submission with the embedding and just require that the hash is unique. This fixes an issue when using Postgres that may generate error messages like "Values larger than 1/3 of a buffer page cannot be indexed.".

# Version 0.5.0

* First release for Pretalx 2025.1.0
* Renamed permissions to align with the new upstream release

# Version 0.4.5

* Also increase version number in code

# Version 0.4.4.

* Correctly sort imports

# Version 0.4.3

* Load numpy and umap on application startup. This makes startups slower and pretalx will consume more memory, but requests will be served faster.

# Version 0.4.2

* Fix latest release, we were still depending on Pretalx main

# Version 0.4.1

* Depend only on the latest stable release of Pretalx

# Version 0.4.0

* Make the plugin only available for users with orga.change_submissions permissions

# Version 0.3.1

* Reformat source code

# Version 0.3.0

* Make the embedding of submissions based on the title, abstract, and description. Previously, the abstract was not used. The embeddings need to be regenerated, so the existing embeddings are deleted when the database is migrated.
* Depend on *umap-learn* instead of *umap*. Previously the project depended on the wrong package.
* Updated the project URL.

# Version 0.2.0

* Reviewers can now indicate their preferences for submissions in text form.
* Based on those preferences, the best matching reviewers are then shown next to the submissions.
* Embeddings that are outdated are now periodically removed from the database.
* Should a Celery job fail to execute, those stall embeddings are periodically removed as well.
* When a submission is saved, the generation of a new embedding vector for this submission will be instantly triggered.

# Version 0.1.0

* Initial version
