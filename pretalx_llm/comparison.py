import hashlib
import heapq
import itertools
import json
import logging

import numpy as np
import umap

from . import cache_utils
from .redislock import PretalxOptionalLock

# from silk.profiling.profiler import silk_profile


logger = logging.getLogger(__name__)


class SubmissionComparison:
    """
    Compare submissions with other submissions based on their embeddings.
    """

    def __init__(self, submissions):
        """
        Initialize a new instance with a list of submissions that will be used as a bases.
        """

        # Hash the json representation so that it can be used as a cache key.
        hasher = hashlib.sha256()
        for submission in submissions:
            j = submission.embeddings_data
            hasher.update(json.dumps(j).encode("utf-8"))
            t = np.array(j)

            # Normalize the embedding vectors
            submission.nbvec = t / np.linalg.norm(t)
        self.submissions = submissions
        logger.info(
            "We have {} submissions in the constructor".format(len(self.submissions))
        )
        self.inputhash = hasher.hexdigest()

    def _compute_distances(self, left, right):
        """
        Compute a distance matrix of the distances between left and right.

        Left and right must be lists of submissions.
        """
        if len(left) == 0:
            return []
        left_v = np.array([x.nbvec for x in left])
        right_v = np.array([x.nbvec for x in right])
        distances = np.dot(left_v, right_v.T)
        return distances

    def _compare_submission_set(self, n, left, submissions, positions):
        """_Find the n most related submissions among submissions for every submission in left._

        Left must be a subset of the submissions in submissions. Positions must be a list of integers of length len(left). The entry at position i must be the index of left[i] in submissions. For example when left[0] can be found at submissions[3], then positions[0] must be 3.

        Args:
            n (int): _number of related submissions to find_
            left (List[Submission]): _List of Submissions to find related ones for (subset of submissions)_
            submissions (List[Submission]): _All submissions that should be compared with_
            positions (List[int]): _list of length len(left) with indexes so that submissions[position[i]] = left[i]_
        """
        distances = self._compute_distances(left, submissions)
        for i in range(len(left)):
            row = distances[i]
            other_idx = itertools.chain(
                range(positions[i]), range(positions[i] + 1, len(submissions))
            )
            # logger.info("Other idx is {}".format(other_idx))
            best_idx = heapq.nlargest(n, other_idx, key=lambda x, d=row: d[x])
            left[i].related = [submissions[x] for x in best_idx]

    # @silk_profile(name="Umap")
    def add_2d_vectors(self, dimensions=2):
        """_Reduce the high dimensional embedding vectors to a lower dimension, such as 2._

        The lower dimensional vectors are then added to the submission objects using the attribute low_dim.

        Args:
            dimensions (int, optional): _the lower dimension_. Defaults to 2.
        """
        if len(self.submissions) == 0:
            return
        # Checck whether the result is already in cache
        key = "umap_{}_{}".format(self.inputhash, dimensions)
        low_dim = cache_utils.maybe_get(key)
        if low_dim is None:
            # Get a lock and check the cache again
            with PretalxOptionalLock("umap_{}_{}".format(self.inputhash, dimensions)):
                low_dim = cache_utils.maybe_get(key)
                if low_dim is None:
                    # We need to recompute, it is really not in the cache
                    logger.info("We have to recompute")
                    high_dim = np.array([x.nbvec for x in self.submissions])
                    # Setting a fixed random state makes this code deterministic but slower
                    reducer = umap.UMAP(n_components=dimensions)  # , random_state=42)
                    low_dim = [
                        [float(x) for x in y] for y in reducer.fit_transform(high_dim)
                    ]
                    cache_utils.maybe_set(key, low_dim)
                else:
                    logger.info("Using cached result")
        else:
            logger.info("Using cached result")

        for idx, submission in enumerate(self.submissions):
            submission.coords = low_dim[idx]

    def rank_with_query(self, query_embedding):
        """
        Get a ranked list of the submissions, ranked for the similarity with the query embedding.
        """
        t = np.array(query_embedding)
        normalized = t / np.linalg.norm(t)

        left_v = np.array([normalized])
        right_v = np.array([x.nbvec for x in self.submissions])
        distances = np.dot(left_v, right_v.T)
        for idx, submission in enumerate(self.submissions):
            submission.hintscore = distances[0][idx]

        return sorted(self.submissions, key=lambda x: x.hintscore, reverse=True)

    def rank_with_reviewed(self, reviewed):
        """
        Get a ranked list of the submissions, based on the similarity with the submissions already reviewed.
        """
        for r in reviewed:
            t = np.array(r.embeddings_data)
            r.nbvec = t / np.linalg.norm(t)
        distances = self._compute_distances(self.submissions, reviewed)
        result = []
        for idx, submission in enumerate(self.submissions):
            row = distances[idx]
            other_idx = [x for x in range(len(reviewed)) if reviewed[x] != submission]
            best_idx = max(other_idx, key=lambda x, row=row: row[x])
            submission.hint = reviewed[best_idx]
            submission.hintscore = row[best_idx]
        result = sorted(self.submissions, key=lambda x: x.hintscore, reverse=True)
        return result

    # @silk_profile(name="Comparison")
    def compare_submissions(self, n, filter, only_same_track=False):
        """
        Compare all submissions against each other, picking the n most similar ones.

        A filter function can be supplied that takes a submission as argument. Only those submissions for which the filter returned true will be compared to other submissions.

        Optionally, only_same_track can be set to true. Then, submissions are only compared with other submissions in the same track.

        Either all or just the filtered submissions will then have a new attribute named related that contains a list of the n most similar submissions.
        """
        left = []

        logger.info("Only same track is {}".format(only_same_track))
        logger.info("Filter is {}".format(filter))

        if only_same_track:
            self._compare_same_track_submissions(n, filter, left)

        else:
            positions = []
            for idx, submission in enumerate(self.submissions):
                if filter(submission):
                    positions.append(idx)
                    left.append(submission)
            logger.debug("We have {} submissions after the filter".format(len(left)))
            self._compare_submission_set(n, left, self.submissions, positions)

        return left

    def _compare_same_track_submissions(self, n, filter, left):
        trackmap = {}
        filtermap = {}
        positionmap = {}
        for submission in self.submissions:
            if submission.track is None:
                # Should the track be None, then we cound it effectively as a separate track
                track_id = ""
            else:
                track_id = str(submission.track.pk)

            local = trackmap.setdefault(track_id, [])

            if filter(submission):
                positionmap.setdefault(track_id, []).append(len(local))
                left.append(submission)
                filtermap.setdefault(track_id, []).append(submission)

            local.append(submission)
        for track_id, trackfiltered in filtermap.items():
            self._compare_submission_set(
                n, trackfiltered, trackmap[track_id], positionmap[track_id]
            )
