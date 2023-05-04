import supervisely as sly
import sly_globals as g
import sly_functions as f
import numpy as np
import os


class TrackerContainer:
    def __init__(self, context, api, logger):
        self.api = api
        self.logger = logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]
        self.stop = len(self.object_ids) * self.frames_count

        self.geometries = []
        self.frames_indexes = []

        self.add_geometries()
        self.add_frames_indexes()
        self.video_path = self.download_video()

        self.logger.info("TrackerController Initialized")

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

    def download_video(self):
        video_info = self.api.video.get_info_by_id(self.video_id)
        video_filename = video_info.name
        save_video_path = os.path.join(g.save_video_dir, video_filename)
        if not sly.fs.file_exists(save_video_path):
            self.api.video.download_path(
                id=self.video_id,
                path=save_video_path,
            )
        return save_video_path

    def track(self):
        for object_id, geometry in zip(self.object_ids, self.geometries):
            frame_start = (
                self.frame_index if self.direction == "forward" else self.frames_indexes[-1]
            )
            frame_end = self.frames_indexes[-1] if self.direction == "forward" else self.frame_index
            if isinstance(geometry, sly.Rectangle):
                points, rect_w, rect_h = f.geometry_to_np(geometry)
            else:
                points = f.geometry_to_np(geometry)
            # input data must consist of points with (time, height, width) order
            input_data = []
            for point in points:
                input_data.append([0, point[1], point[0]])
            input_data = np.array(input_data).astype(np.int32)
            tracked_points, input_height, input_width = f.run_model(
                self.video_path, frame_start, frame_end, input_data, self.direction
            )
            tracked_points = f.check_bounds(tracked_points, input_height, input_width)
            for i in range(self.frames_count):
                frame_index = self.frames_indexes[i + 1]
                new_points = tracked_points[:, i]
                if isinstance(geometry, sly.Rectangle):
                    new_figure = f.np_to_geometry(
                        new_points, geometry.geometry_name(), rect_w, rect_h
                    )
                else:
                    new_figure = f.np_to_geometry(new_points, geometry.geometry_name())
                self.api.video.figure.create(
                    self.video_id,
                    object_id,
                    frame_index,
                    new_figure.to_json(),
                    new_figure.geometry_name(),
                    self.track_id,
                )
                cur_pos = i + frame_start + 1
                stop = self._notify(cur_pos)
                if stop:
                    self.logger.info("Task stoped by user")
                    self._notify(self.stop)
                    return
        self.logger.info("Tracking completed")
        self._notify(self.stop)

    def _notify(self, pos: int):
        return self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            min(self.frames_indexes),
            max(self.frames_indexes),
            pos,
            self.stop,
        )
