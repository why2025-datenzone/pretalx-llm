import logging

from pretalx.submission.models.submission import Submission

logger = logging.getLogger(__name__)


class SoftwareFilterer:
    """
    This class filters a list of Submission objects directly in the application based on a filter provided by an SubmissionFilterForm instead of filtering them using SQL statements as it is done by Pretalx at the moment.

    The main use case is picking out only certain submissions from a longer list that already been retrieved from the database, such as when you only want to display a subset of these submissions, but you need more submissions anyway, such as when you want to compare the smaller set of submissions with all submissions retrieved from the database.
    """

    def __init__(
        self, filter_form, user, review_status_form=None, can_search_speakers=False
    ):
        self.submission_type = filter_form.cleaned_data.get("submission_type")
        self.content_locale = filter_form.cleaned_data.get("content_locale")
        self.track = filter_form.cleaned_data.get("track")
        self.tags = filter_form.cleaned_data.get("tags")
        self.query = filter_form.cleaned_data.get("q")
        self.pending_is_null = filter_form.cleaned_data.get("pending_state__isnull")
        self.filter_state = []
        self.filter_pending_state = []
        self.review_status_form = review_status_form
        self.user = (user,)
        states = filter_form.cleaned_data.get("state")
        if states:
            for state in states:
                if state.startswith("pending_state__"):
                    self.filter_pending_state.append(state.split("__")[1])
                else:
                    self.filter_state.append(state)
        self.can_search_speakers = can_search_speakers

    def prefetch(self, queryset):
        """
        The SubmissionFilterForm supports filtering based on speaker names, which only works when the speaker names have been prefetched. You can call this method with a queryset and it will return this queryset with the speakers prefetched.
        """
        if self.query and self.can_search_speakers:
            queryset = queryset.prefetch_related("speakers")
        return queryset

    def filter(self, submission: Submission):
        """
        Performs the actual filtering of a submission, it will return True when the submission matches the filter, False otherwiese.
        """
        if self.submission_type and (
            submission.submission_type not in self.submission_type
        ):
            return False
        if self.content_locale and (
            submission.content_locale not in self.content_locale
        ):
            return False
        if self.track and (submission.track not in self.track):
            return False
        if (
            self._filter_query(submission)
            and self._filter_review_status(submission)
            and self._filter_state(submission)
        ):
            return True
        else:
            return False

    def _filter_review_status(self, submission: Submission):
        if self.review_status_form is not None:
            if self.review_status_form.get_review_status() == 1:
                if submission.hasreviewed is False:
                    return False
            elif self.review_status_form.get_review_status() == 2:
                if submission.hasreviewed is True:
                    return False
        return True

    def _filter_state(self, submission: Submission):
        if (len(self.filter_state) > 0 or len(self.filter_pending_state) > 0) and not (
            submission.pending_state in self.filter_pending_state
            or submission.state in self.filter_state
        ):
            return False
        if self.pending_is_null and submission.state is not None:
            return False
        return True

    def _filter_query(self, submission: Submission):
        if self.query and not (
            self.query.lower() in submission.title.lower()
            or self.query.lower() in submission.code.lower()
            or (
                self.can_search_speakers
                and any(
                    [
                        self.query.lower() in x.name.lower()
                        for x in submission.speakers.all()
                    ]
                )
            )
        ):
            return False
        return True
