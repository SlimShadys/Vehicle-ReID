import numpy as np

class Trajectory():
    def __init__(self, id, tracks, n_cams):
        self.id = id
        self.tracks = tracks # This is at least one Track object
        self.n_cams = n_cams

        self.start_time = None
        self.end_time = None
        self._mean_feature = None
        self._cams = None

    def merge_with(self, trajectory):
        """Merge an other multicam tracklet to this one."""
        self.tracks.extend(trajectory.tracks)
        self._mean_feature = None
        if self._cams is not None:
            for track in trajectory.tracks:
                self._cams |= 1 << track.cam

    @property
    def mean_feature(self):
        """Mean feature of all single cam tracklets."""
        if self._mean_feature is None:
            self._mean_feature = np.zeros_like(self.tracks[0]._mean_feature)
            for track in self.tracks:
                self._mean_feature += track._mean_feature
            self._mean_feature /= np.linalg.norm(self._mean_feature)
        return self._mean_feature

    @property
    def cams(self):
        """Camera occurrence bitmap."""
        if self._cams is None:
            self._cams = 0
            for track in self.tracks:
                self._cams |= 1 << track.cam
        return self._cams

class Track():
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id

        self.id = None
        self.cam = None
        self.start_time = None
        self.end_time = None
        self._mean_feature = None
        self.frames = {}
        self.bboxes = []
        self.features = []

    # tlwh format
    def extract_bboxes(self):
        try:
            self.bboxes = [f[1] for f in self.frames.values()]
        except:
            self.bboxes = [f['bounding_box'] for f in self.frames.values()]

    def extract_features(self):
        try:
            self.features = [f[2] for f in self.frames.values()]
        except:
            self.features = [f['features'] for f in self.frames.values()]

    def mean_feature(self, method="area_avg"):
        if self._mean_feature is None:
            if len(self.frames) == 0:
                return self._mean_feature
            else:
                self._mean_feature = np.zeros_like(self.features[0])
                if method == "area_avg":
                    div = min(map(lambda x: x[2] * x[3], self.bboxes))

                for i, f in enumerate(self.features):
                    if method == "area_avg":
                        area = self.bboxes[i][2] * self.bboxes[i][3]
                        self._mean_feature += np.array(f) * (area / div)
                    else:
                        self._mean_feature += np.array(f)

                self._mean_feature /= np.linalg.norm(self._mean_feature)
        return self._mean_feature

    def get_track_id(self):
        return self.id
    
    def get_vehicle_id(self):
        return self.vehicle_id
    
    def get_cam_id(self):
        return self.cam

    def add_frames(self, frames):
        self.frames = frames

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def __str__(self):
        return f"Track ID: {self.id}"