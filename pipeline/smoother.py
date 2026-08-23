from collections import defaultdict, deque, Counter
from typing import Dict, Any


class AttributeSmoother:

    def __init__(
        self,
        window_size: int = 5,
        max_missing_frames: int = 30
    ):
        self.window_size = window_size
        self.max_missing_frames = max_missing_frames

        self.history = defaultdict(
            lambda: defaultdict(
                lambda: deque(
                    maxlen=self.window_size
                )
            )
        )

        self.missing_frames = defaultdict(int)

    def update(
        self,
        track_id: int,
        attributes: Dict[str, Any]
    ) -> Dict[str, Any]:

        if track_id < 0:
            return attributes

        self.missing_frames[track_id] = 0

        if not attributes:
            return {}

        smoothed = {}

        for attribute_name, value in attributes.items():

            self.history[
                track_id
            ][
                attribute_name
            ].append(value)

            smoothed[
                attribute_name
            ] = self._majority_vote(
                self.history[
                    track_id
                ][
                    attribute_name
                ]
            )

        return smoothed

    def _majority_vote(self, values):

        if not values:
            return None

        counter = Counter(values)

        return counter.most_common(1)[0][0]

    def mark_missing(
        self,
        active_track_ids: set[int]
    ):

        all_track_ids = set(
            self.history.keys()
        )

        for track_id in all_track_ids:

            if track_id in active_track_ids:
                self.missing_frames[track_id] = 0

            else:
                self.missing_frames[track_id] += 1

        self._remove_old_tracks()

    def _remove_old_tracks(self):

        remove_ids = []

        for track_id, missing_count in self.missing_frames.items():

            if missing_count > self.max_missing_frames:
                remove_ids.append(track_id)

        for track_id in remove_ids:

            self.history.pop(
                track_id,
                None
            )

            self.missing_frames.pop(
                track_id,
                None
            )

    def reset(self):

        self.history.clear()
        self.missing_frames.clear()