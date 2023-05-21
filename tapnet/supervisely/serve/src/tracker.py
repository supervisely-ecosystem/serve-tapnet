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

        self.pbar_value = 0
        self.first_part = round(self.stop * 0.5)
        self.second_part = self.stop - self.first_part

        self.geometries = []
        self.frames_indexes = []

        self.add_geometries()
        self.add_frames_indexes()
        self.load_frames()

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

    def load_frames(self):
        frame_pbar_unit = self.first_part / len(self.frames_indexes)
        rgbs = []
        for frame_index in self.frames_indexes:
            self.pbar_value += frame_pbar_unit
            img_rgb = self.api.video.frame.download_np(self.video_id, frame_index)
            rgbs.append(img_rgb)
            self._notify(self.pbar_value)
        self.frames = np.stack(rgbs)

    def track(self):
        for pos, (object_id, geometry) in enumerate(zip(self.object_ids, self.geometries), start=1):
            frame_start = (
                self.frame_index if self.direction == "forward" else self.frames_indexes[-1]
            )
            frame_end = self.frames_indexes[-1] if self.direction == "forward" else self.frame_index
            if isinstance(geometry, sly.Rectangle):
                points, rect_w, rect_h = f.geometry_to_np(geometry)
            elif isinstance(geometry, sly.GraphNodes):
                points = f.geometry_to_np(geometry)
                video_info = self.api.video.get_info_by_id(self.video_id)
                project_id = video_info.project_id
                project_meta_json = self.api.project.get_meta(project_id)
                object_info = self.api.video.object.get_info_by_id(object_id)
                class_id = object_info.class_id
                for cls in project_meta_json["classes"]:
                    if cls["id"] == class_id:
                        geometry_config = cls["geometry_config"]
                        break
            else:
                points = f.geometry_to_np(geometry)
            # input data must consist of points with (time, height, width) order
            input_data = []
            for point in points:
                input_data.append([0, point[1], point[0]])
            input_data = np.array(input_data).astype(np.int32)
            tracked_points, input_height, input_width = f.run_model(
                self.frames, frame_start, frame_end + 1, input_data, self.direction
            )
            tracked_points = f.check_bounds(tracked_points, input_height, input_width)

            figure_pbar_unit = (self.second_part / self.frames_count) / len(self.object_ids)

            for i in range(self.frames_count):
                frame_index = self.frames_indexes[i + 1]
                new_points = tracked_points[:, i + 1]
                if isinstance(geometry, sly.Rectangle):
                    new_figure = f.np_to_geometry(
                        new_points, geometry.geometry_name(), rect_w, rect_h
                    )
                elif isinstance(geometry, sly.GraphNodes):
                    new_figure_json = f.get_graph_json(new_points, geometry_config)
                    geometry_name = "graph"
                else:
                    new_figure = f.np_to_geometry(new_points, geometry.geometry_name())
                if not isinstance(geometry, sly.GraphNodes):
                    geometry_name = new_figure.geometry_name()
                    new_figure_json = new_figure.to_json()
                self.api.video.figure.create(
                    self.video_id,
                    object_id,
                    frame_index,
                    new_figure_json,
                    geometry_name,
                    self.track_id,
                )
                self.pbar_value += figure_pbar_unit
                stop = self._notify(self.pbar_value)
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
