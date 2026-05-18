"""
CARLA Autonomous Vehicle Dataset Collector
R-Sync Paper — Multi-Town Data Collection

Usage:
    python collect_data.py --town Town01
    python collect_data.py --town Town07   # held-out test set
    python collect_data.py --town Town04 --frames 5000 --fps 20

Collects 5000 frames of multi-modal sensor data per town.
Output saved to: data/{TOWN}/driving_log.csv + sensor folders

Towns used in paper:
    Training: Town01, Town02, Town03, Town04, Town05, Town06, Town10HD
    Test set: Town07 (held-out, no Dirichlet partitioning)
"""

import argparse
import carla
import numpy as np
import cv2
import os
import random
import time
import pandas as pd
from tqdm import tqdm

# ── CLI ARGUMENTS ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="CARLA multi-modal dataset collector for R-Sync paper"
)
parser.add_argument(
    "--town", type=str, default="Town01",
    help="CARLA town name (e.g. Town01, Town07)"
)
parser.add_argument(
    "--frames", type=int, default=5000,
    help="Number of frames to collect (default: 5000)"
)
parser.add_argument(
    "--fps", type=int, default=20,
    help="Simulation FPS (default: 20)"
)
parser.add_argument(
    "--output", type=str, default=None,
    help="Output directory (default: data/{town})"
)
parser.add_argument(
    "--host", type=str, default="localhost",
    help="CARLA server host (default: localhost)"
)
parser.add_argument(
    "--port", type=int, default=2000,
    help="CARLA server port (default: 2000)"
)
cli_args = parser.parse_args()

# ── CONFIG ────────────────────────────────────────────────────────────────────
TOWN       = cli_args.town
NUM_FRAMES = cli_args.frames
FPS        = cli_args.fps
HOST       = cli_args.host
PORT       = cli_args.port
OUTPUT_DIR = cli_args.output if cli_args.output else os.path.join("data", TOWN)

MAX_DEPTH  = 80.0   # clip depth to driving-relevant range (meters)
MIN_DEPTH  = 0.1    # minimum valid depth (meters)

CITYSCAPES_PALETTE = np.array([
    [0,0,0],[128,64,128],[244,35,232],[70,70,70],[102,102,156],
    [190,153,153],[153,153,153],[250,170,30],[220,220,0],
    [107,142,35],[152,251,152],[70,130,180],[220,20,60],
    [255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],
    [0,0,230],[119,11,32]
], dtype=np.uint8)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_actor_blueprints(world, filter_str, generation):
    bps = world.get_blueprint_library().filter(filter_str)
    if generation.lower() == "all":
        return bps
    try:
        generation = int(generation)
        return [bp for bp in bps
                if int(bp.get_attribute("generation")) == generation]
    except Exception:
        return bps


# ── COLLECTOR ─────────────────────────────────────────────────────────────────
class Collector:
    def __init__(self):
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(90.0)

        print(f"Loading world: {TOWN} ...")
        self.world = self.client.load_world(TOWN)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_random_device_seed(42)

        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 1.0 / FPS
        settings.no_rendering_mode   = False
        self.world.apply_settings(settings)

        self.bp_lib        = self.world.get_blueprint_library()
        self.vehicle       = None
        self.sensors       = {}
        self.vehicles_list = []
        self.walkers_list  = []

        self.gnss_data      = None
        self.imu_data       = None
        self.collision_flag = 0
        self.lidar_points   = None
        self.radar_data     = None

        self.current_frame = dict(rgb=None, depth=None, seg=None,
                                  timestamp=None)
        self.frame_log = []

        self._create_dirs()
        self._set_weather()

        for _ in range(20):
            self.world.tick()

    # ── Directory setup ───────────────────────────────────────────────────────
    def _create_dirs(self):
        for d in ["rgb", "rgb_npy", "depth_raw", "depth_gt", "depth_mask",
                  "depth_normalized", "depth_colormap",
                  "segmentation", "segmentation_npy"]:
            os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

    def _set_weather(self):
        self.world.set_weather(carla.WeatherParameters(
            cloudiness=10, precipitation=0, sun_altitude_angle=70
        ))

    # ── Spawn ─────────────────────────────────────────────────────────────────
    def _spawn_vehicle_and_sensors(self):
        bp     = self.bp_lib.filter("vehicle.tesla.model3")[0]
        spawn  = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn)

        cam_tf = carla.Transform(
            carla.Location(x=-6, z=2.5),
            carla.Rotation(pitch=-10)
        )

        # GNSS
        gnss_bp = self.bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", str(1.0 / FPS))
        self.gnss = self.world.spawn_actor(
            gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.gnss.listen(self._process_gnss)

        # IMU
        imu_bp = self.bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(1.0 / FPS))
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.imu.listen(self._process_imu)

        # Collision
        col_bp = self.bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._process_collision)

        # LiDAR
        lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("points_per_second", "10000")
        lidar_bp.set_attribute("rotation_frequency", str(FPS))
        self.lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(), attach_to=self.vehicle)
        self.lidar.listen(self._process_lidar)

        # Radar
        radar_bp = self.bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", "30")
        radar_bp.set_attribute("vertical_fov", "20")
        radar_bp.set_attribute("range", "50")
        self.radar = self.world.spawn_actor(
            radar_bp, carla.Transform(), attach_to=self.vehicle)
        self.radar.listen(self._process_radar)

        # Cameras (RGB, Depth, Semantic Segmentation)
        def spawn_camera(name, callback):
            bp = self.bp_lib.find(f"sensor.camera.{name}")
            bp.set_attribute("image_size_x", "1024")
            bp.set_attribute("image_size_y", "512")
            bp.set_attribute("fov", "90")
            bp.set_attribute("sensor_tick", str(1.0 / FPS))
            sensor = self.world.spawn_actor(
                bp, cam_tf, attach_to=self.vehicle)
            sensor.listen(callback)
            self.sensors[name] = sensor

        spawn_camera("rgb", self._process_rgb)
        spawn_camera("depth", self._process_depth)
        spawn_camera("semantic_segmentation", self._process_seg)

        for _ in range(10):
            self.world.tick()

    def _spawn_traffic(self):
        blueprints   = get_actor_blueprints(
            self.world, "vehicle.*", "all")
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        batch = []
        for transform in spawn_points[:40]:
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            batch.append(
                carla.command.SpawnActor(bp, transform).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor, True,
                        self.tm.get_port()))
            )
        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                self.vehicles_list.append(r.actor_id)

        # Walkers
        walker_bps = get_actor_blueprints(
            self.world, "walker.pedestrian.*", "all")
        batch = []
        for _ in range(20):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                bp = random.choice(walker_bps)
                batch.append(carla.command.SpawnActor(
                    bp, carla.Transform(loc)))
        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                self.walkers_list.append(r.actor_id)

        print(f"Spawned {len(self.vehicles_list)} vehicles, "
              f"{len(self.walkers_list)} walkers")

    # ── Sensor callbacks ──────────────────────────────────────────────────────
    def _process_rgb(self, image):
        self.current_frame["rgb"] = np.frombuffer(
            image.raw_data, np.uint8
        ).reshape(image.height, image.width, 4)[:, :, :3]
        self.current_frame["timestamp"] = image.timestamp

    def _process_depth(self, image):
        arr = np.frombuffer(
            image.raw_data, dtype=np.uint8
        ).reshape(image.height, image.width, 4)
        B = arr[:, :, 0].astype(np.float32)
        G = arr[:, :, 1].astype(np.float32)
        R = arr[:, :, 2].astype(np.float32)
        normalized   = (R + G * 256.0 + B * 65536.0) / (256.0**3 - 1.0)
        depth_meters = normalized * 1000.0
        self.current_frame["depth"] = depth_meters

    def _process_seg(self, image):
        arr = np.frombuffer(
            image.raw_data, np.uint8
        ).reshape(image.height, image.width, 4)
        self.current_frame["seg"] = arr[:, :, 2]

    def _process_gnss(self, data):
        self.gnss_data = data

    def _process_imu(self, data):
        self.imu_data = data

    def _process_collision(self, event):
        self.collision_flag = 1

    def _process_lidar(self, data):
        pts = np.frombuffer(
            data.raw_data, dtype=np.float32
        ).reshape(-1, 4)
        self.lidar_points = pts

    def _process_radar(self, data):
        self.radar_data = [(d.depth, d.velocity) for d in data]

    def _get_min_obstacle_distance(self):
        if self.lidar_points is not None and len(self.lidar_points) > 0:
            return float(np.min(
                np.linalg.norm(self.lidar_points[:, :3], axis=1)
            ))
        if self.radar_data:
            return float(min(d[0] for d in self.radar_data))
        return np.nan

    # ── Main collection loop ──────────────────────────────────────────────────
    def collect(self):
        self._spawn_vehicle_and_sensors()
        self._spawn_traffic()
        self.vehicle.set_autopilot(True)

        print(f"\nCollecting {NUM_FRAMES} frames from {TOWN} ...")
        print(f"Output: {OUTPUT_DIR}")
        pbar     = tqdm(total=NUM_FRAMES)
        frame_id = 0

        while frame_id < NUM_FRAMES:
            self.world.tick()

            if any(self.current_frame[k] is None
                   for k in ["rgb", "depth", "seg"]):
                continue

            fname     = f"{frame_id:06d}"
            depth_raw = self.current_frame["depth"]
            rgb       = self.current_frame["rgb"]
            seg       = self.current_frame["seg"]

            # RGB
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "rgb", f"{fname}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
            np.save(os.path.join(OUTPUT_DIR, "rgb_npy", f"{fname}.npy"), rgb)

            # Depth — raw
            np.save(
                os.path.join(OUTPUT_DIR, "depth_raw", f"{fname}.npy"),
                depth_raw.astype(np.float32)
            )

            # Depth — validity mask
            valid_mask = (depth_raw >= MIN_DEPTH) & (depth_raw <= MAX_DEPTH)
            np.save(
                os.path.join(OUTPUT_DIR, "depth_mask", f"{fname}.npy"),
                valid_mask.astype(np.uint8)
            )

            # Depth — ground truth (clipped)
            depth_clipped = np.clip(depth_raw, MIN_DEPTH, MAX_DEPTH)
            np.save(
                os.path.join(OUTPUT_DIR, "depth_gt", f"{fname}.npy"),
                depth_clipped.astype(np.float32)
            )

            # Depth — log-normalized (for neural networks)
            depth_norm = (np.log(depth_clipped + 1.0) /
                          np.log(MAX_DEPTH + 1.0))
            np.save(
                os.path.join(OUTPUT_DIR, "depth_normalized", f"{fname}.npy"),
                depth_norm.astype(np.float32)
            )

            # Depth — colormap visualization
            depth_vis     = cv2.bilateralFilter(
                depth_clipped.astype(np.float32), 5, 50, 50)
            depth_vis_inv = 1.0 - (depth_vis - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            depth_uint8   = (np.clip(depth_vis_inv, 0, 1) * 255).astype(np.uint8)
            depth_color   = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            depth_color[~valid_mask] = [0, 0, 0]
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "depth_colormap", f"{fname}.png"),
                depth_color
            )

            # Segmentation
            np.save(
                os.path.join(OUTPUT_DIR, "segmentation_npy", f"{fname}.npy"),
                seg
            )
            seg_vis = CITYSCAPES_PALETTE[np.clip(
                seg, 0, len(CITYSCAPES_PALETTE) - 1)]
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "segmentation", f"{fname}.png"),
                cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)
            )

            # Telemetry
            velocity  = self.vehicle.get_velocity()
            speed_kmh = 3.6 * np.linalg.norm(
                [velocity.x, velocity.y, velocity.z])
            control   = self.vehicle.get_control()
            transform = self.vehicle.get_transform()
            location  = transform.location
            rotation  = transform.rotation

            gps_lat = self.gnss_data.latitude  if self.gnss_data else np.nan
            gps_lon = self.gnss_data.longitude if self.gnss_data else np.nan
            gps_alt = self.gnss_data.altitude  if self.gnss_data else np.nan

            accel = (self.imu_data.accelerometer
                     if self.imu_data else carla.Vector3D())
            gyro  = (self.imu_data.gyroscope
                     if self.imu_data else carla.Vector3D())

            lidar_mean = (
                float(np.mean(np.linalg.norm(
                    self.lidar_points[:, :3], axis=1)))
                if self.lidar_points is not None
                and len(self.lidar_points) > 0 else np.nan
            )

            if self.radar_data:
                radar_mean_dist = float(
                    np.mean([d[0] for d in self.radar_data]))
                radar_mean_vel  = float(
                    np.mean([d[1] for d in self.radar_data]))
            else:
                radar_mean_dist = radar_mean_vel = np.nan

            valid_depth      = depth_clipped[valid_mask]
            depth_mean_valid = (float(np.mean(valid_depth))
                                if valid_depth.size > 0 else np.nan)

            self.frame_log.append({
                "frame_id":            frame_id,
                "timestamp":           self.current_frame["timestamp"],
                "speed_kmh":           speed_kmh,
                "steering_angle":      control.steer,
                "throttle":            control.throttle,
                "brake":               control.brake,
                "distance_to_object":  self._get_min_obstacle_distance(),
                "position_x":          location.x,
                "position_y":          location.y,
                "position_z":          location.z,
                "rotation_yaw":        rotation.yaw,
                "rotation_pitch":      rotation.pitch,
                "rotation_roll":       rotation.roll,
                "gps_latitude":        gps_lat,
                "gps_longitude":       gps_lon,
                "gps_altitude":        gps_alt,
                "accel_x":             accel.x,
                "accel_y":             accel.y,
                "accel_z":             accel.z,
                "gyro_x":              gyro.x,
                "gyro_y":              gyro.y,
                "gyro_z":              gyro.z,
                "lidar_mean_distance": lidar_mean,
                "radar_mean_distance": radar_mean_dist,
                "radar_mean_velocity": radar_mean_vel,
                "depth_mean":          depth_mean_valid,
                "collision_flag":      self.collision_flag,
            })

            self.collision_flag  = 0
            self.current_frame   = dict(
                rgb=None, depth=None, seg=None, timestamp=None)
            frame_id += 1
            pbar.update(1)

        pbar.close()

        log_path = os.path.join(OUTPUT_DIR, "driving_log.csv")
        pd.DataFrame(self.frame_log).to_csv(log_path, index=False)
        print(f"\n✅ Done! {NUM_FRAMES} frames saved to: {OUTPUT_DIR}")
        print(f"   CSV log: {log_path}")

        self._cleanup()

    def _cleanup(self):
        for s in ["gnss", "imu", "lidar", "radar", "collision_sensor"]:
            sensor = getattr(self, s, None)
            if sensor:
                sensor.destroy()
        for sensor in self.sensors.values():
            if sensor:
                sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        self.client.apply_batch([
            carla.command.DestroyActor(x)
            for x in self.vehicles_list + self.walkers_list
        ])


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    collector = Collector()
    collector.collect()
