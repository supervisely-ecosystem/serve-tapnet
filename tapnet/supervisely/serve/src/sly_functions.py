from typing import Mapping

import jax
import haiku as hk
import numpy as np
import tensorflow as tf
import mediapy as media

from tapnet.supervised_point_prediction import SupervisedPointPrediction
from tapnet import tapnet_model
from tapnet import task
from tapnet.configs.tapnet_config import get_config
import sly_globals as g
from supervisely.geometry.geometry import Geometry
import supervisely as sly


def _construct_shared_modules(config) -> Mapping[str, task.SharedModule]:
    """Constructs the TAPNet module which is used for all tasks.
    More generally, these are Haiku modules that are passed to all tasks so that
    weights are shared across tasks.
    Returns:
    A dict with a single key 'tapnet_model' containing the tapnet model.
    """
    shared_module_constructors = {
        "tapnet_model": tapnet_model.TAPNet,
    }
    shared_modules = {}

    for shared_mod_name in config.shared_modules.shared_module_names:
        ctor = shared_module_constructors[shared_mod_name]
        kwargs = config.shared_modules[shared_mod_name + "_kwargs"]
        shared_modules[shared_mod_name] = ctor(**kwargs)
    return shared_modules


config = get_config().experiment_kwargs.config
point_prediction = SupervisedPointPrediction(config, **config.supervised_point_prediction_kwargs)


def forward(*args, **kwargs):
    shared_modules = _construct_shared_modules(config)
    return point_prediction.forward_fn(
        *args,
        shared_modules=shared_modules,
        is_training=False,
        **kwargs,
    )


def run_model(input_video_path, frame_start, frame_end, query_points, direction):
    """Reads input video with query points and applies model to input data
    Returns:
    A np.array of tracked points
    """
    if sly.is_production():
        tf.config.experimental.set_visible_devices([], "GPU")
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    transform = hk.transform_with_state(forward)
    forward_fn = transform.apply
    jaxline_mode = "eval_inference"
    random_seed = 42
    eval_rng = jax.random.PRNGKey(random_seed)
    with tf.io.gfile.GFile(
        g.checkpoint_path,
        "rb",
    ) as fp:
        ckpt_state = np.load(fp, allow_pickle=True).item()
        state = ckpt_state["state"]
        params = ckpt_state["params"]
        global_step = ckpt_state["global_step"]
    input_key = "kubric"
    config_inference = config.inference
    config_inference.resize_height = 256
    config_inference.resize_width = 256
    resize_height, resize_width = config_inference.resize_height, config_inference.resize_width
    video = media.read_video(input_video_path)
    video = video[frame_start:frame_end]
    if direction == "backward":
        video = video[::-1]
    input_height, input_width = video.shape[1], video.shape[2]
    height_scaler, width_scaler = input_height / resize_height, input_width / resize_width
    query_points[:, 1] = query_points[:, 1] / height_scaler
    query_points[:, 2] = query_points[:, 2] / width_scaler
    num_frames = frame_end - frame_start
    video = media.resize_video(video, (resize_height, resize_width))
    video = video.astype(np.float32) / 255 * 2 - 1
    num_points = len(query_points)
    occluded = np.zeros((num_points, num_frames), dtype=np.float32)
    inputs = {
        input_key: {
            "video": video[np.newaxis],
            "query_points": query_points[np.newaxis],
            "occluded": occluded[np.newaxis],
        }
    }
    r = forward_fn(
        params=params,
        state=state,
        rng=eval_rng,
        inputs=inputs,
        input_key=input_key,
        get_query_feats=False,
    )
    tracks = r[0]["tracks"][0]
    # convert jax array to numpy array and rescale tracked points back to original size
    tracks = np.array(tracks)
    tracks[:, :, 0] = tracks[:, :, 0] * width_scaler
    tracks[:, :, 1] = tracks[:, :, 1] * height_scaler
    tracks = np.rint(tracks)
    tracks = tracks.astype(np.int32)
    return tracks, input_height, input_width


def geometry_to_np(figure: Geometry):
    if isinstance(figure, sly.Rectangle):
        center = figure.center
        x, y = center.col, center.row
        width, height = figure.width, figure.height
        return np.array([[x, y]]), width, height
    if isinstance(figure, sly.Point):
        return np.array([[figure.col, figure.row]])
    if isinstance(figure, sly.Polygon):
        return figure.exterior_np[:, ::-1].copy()
    if isinstance(figure, sly.GraphNodes):
        nodes = figure.nodes  # dict (str - sly.Node)
        nodes = [node for node in nodes.values()]  # [sly.Node]
        point_locations = [node.location for node in nodes]  # [sly.Pointlocation]
        points = [[pl.col, pl.row] for pl in point_locations]  # [[x, y]]
        return np.array(points)
    raise ValueError(f"Can't process figures with type `{figure.geometry_name()}`")


def np_to_geometry(
    points: np.ndarray, geom_type: str, rect_w: int = None, rect_h: int = None
) -> Geometry:
    if geom_type == "rectangle":
        center_x, center_y = points.squeeze().astype(int)
        top = center_y - (rect_h / 2)
        left = center_x - (rect_w / 2)
        bottom = center_y + (rect_h / 2)
        right = center_x + (rect_w / 2)
        fig = sly.Rectangle(top, left, bottom, right)
        return fig
    if geom_type == "point":
        col, row = points.squeeze().astype(int)
        return sly.Point(row, col)
    if geom_type == "polygon":
        obj = points.astype(int)[:, ::-1]
        exterior = [sly.PointLocation(*obj_point) for obj_point in obj]
        return sly.Polygon(exterior=exterior)
    raise ValueError(f"Can't process figures with type `{geom_type}`")


def check_bounds(points: np.ndarray, h_max: int, w_max: int):
    points[:, 0] = np.clip(points[:, 0], a_max=w_max - 1, a_min=0)
    points[:, 1] = np.clip(points[:, 1], a_max=h_max - 1, a_min=0)
    return points


def get_graph_json(new_points, geometry_config):
    graph_json = geometry_config
    i = 0
    for node in geometry_config["nodes"].keys():
        col, row = new_points[i]
        graph_json["nodes"][node]["loc"] = [int(col), int(row)]
        i += 1
    return graph_json
