#!/usr/bin/env bash
# Record a rosbag for debugging split tiles, backtracking and walking back through curtains.
#
# Start it BEFORE (or right after) the explorer launch, so the latched topics (/map, /tile_map, /tf_static, the
# mission mode, curtain points) are caught. Stop with Ctrl-C.
#
# What the bag answers:
#   which tile became the last split, and with which options    /mission_manager/events  ('split')   /tile_map ('last_split')
#   why backtracking started / what it did                       /mission_manager/events  ('backtracking', 'backtracked',
#                                                                'backtracking_failed', 'mode', 'curtain')   /rosout
#   the phase it was in (to_split / to_curtain / cross / ...)    /mission_manager/mode, /mission_manager/status
#   whether the curtain virtual walls were really lifted         /virtual_walls (cloud) + /global_costmap/costmap(_updates)
#   whether Nav2 planned / drove / gave up                       /plan, /mission_manager/current_goal, /route, action status
#   the walk through the curtain (direct velocity)               /cmd_vel, /odom, /tf
#   what the lidar / curtain detector saw                        /scan, /curtain/points, /curtain/markers
#
# Usage:
#   ./record_backtracking_bag.sh                     record to <repo>/bags/sim/backtracking_<time>
#   DEPTH=1 ./record_backtracking_bag.sh             also the depth + colour images and camera_info (curtain detector input; large)
#   CLOUD=1 ./record_backtracking_bag.sh             also the raw lidar point cloud (large)
#
# Settings (environment variables):
#   OUT_DIR      where bags go                 (default <repo>/bags/sim, next to this scripts/ folder)
#   NAME         bag name                      (default backtracking_<time>)
#   STORAGE      sqlite3 | mcap                (default sqlite3; mcap needs ros-$ROS_DISTRO-rosbag2-storage-mcap)
#   MAX_SIZE_MB  split the bag at this size    (default 0 = never)
#   DEPTH_TOPIC  (default /camera/depth/image_raw)     CLOUD_TOPIC (default /livox/points)
#   ROS_DISTRO   (default humble)              WS (default ~/2d-challenge)
#
# Replay:
#   ros2 bag play <bag> --clock          (launch RViz with use_sim_time:=true; add the MarkerArray topics
#                                         /mission_manager/tile_markers and /curtain/markers, /virtual_walls,
#                                         /global_costmap/costmap, /plan)
#   ros2 bag play <bag> --topics /mission_manager/events     | jq   (with `ros2 topic echo` in another shell)
# The mission manager also writes every event to <output_dir>/run_<time>/events.jsonl: keep it with the bag.
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
WS="${WS:-$HOME/2d-challenge}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # the 2d-challenge checkout this script lives in
OUT_DIR="${OUT_DIR:-$REPO_DIR/bags/sim}"
NAME="${NAME:-backtracking_$(date +%Y%m%d_%H%M%S)}"
STORAGE="${STORAGE:-sqlite3}"
MAX_SIZE_MB="${MAX_SIZE_MB:-0}"
DEPTH="${DEPTH:-0}"
CLOUD="${CLOUD:-0}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/depth/image_raw}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/livox/points}"
COLOUR_TOPIC="${COLOUR_TOPIC:-/camera/color/image_raw/compressed}"
COLOUR_INFO_TOPIC="${COLOUR_INFO_TOPIC:-/camera/color/camera_info}"

set +u
# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO/setup.bash"
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

TOPICS=(
    # time, transforms, logs (every "Backtracking: ..." / "Curtain: ..." line of the mission manager is on /rosout)
    /clock /tf /tf_static /rosout
    # mission manager
    /mission_manager/mode /mission_manager/status /mission_manager/events
    /mission_manager/current_goal /mission_manager/next_target /mission_manager/route
    /mission_manager/tile_markers /mission_manager/markers /mission_manager/terrain_markers
    /tile_map /tile_report /exploration_complete
    /virtual_walls /cmd_vel /spin_crawl /spin_flat
    # curtains
    /curtain/points /curtain/markers
    # map, costmaps (the updates topics carry the changes between full costmaps), plans
    /map /map_metadata
    /global_costmap/costmap /global_costmap/costmap_updates /global_costmap/published_footprint
    /local_costmap/costmap /local_costmap/costmap_updates
    /plan /local_plan /received_global_plan
    # Nav2 actions: goal status and feedback (why a goal was aborted shows up in /rosout)
    /navigate_to_pose/_action/status /navigate_to_pose/_action/feedback
    /navigate_through_poses/_action/status /navigate_through_poses/_action/feedback
    /follow_waypoints/_action/status /follow_waypoints/_action/feedback
    /compute_path_to_pose/_action/status /spin/_action/status
    # robot
    /scan /odom /terrain_kind /terrain_classifier/details /animation_active
)
[[ "$DEPTH" == 1 ]] && TOPICS+=("$DEPTH_TOPIC" "${DEPTH_TOPIC%/*}/camera_info" "$COLOUR_TOPIC" "$COLOUR_INFO_TOPIC")
[[ "$CLOUD" == 1 ]] && TOPICS+=("$CLOUD_TOPIC")

# Latched topics: record them transient_local so a bag started after the publisher still gets the last message.
QOS_FILE="$(mktemp "${TMPDIR:-/tmp}/lite3_bag_qos.XXXXXX")"
trap 'rm -f "$QOS_FILE"' EXIT
for t in /tf_static /map /tile_map /mission_manager/mode /mission_manager/current_goal /mission_manager/next_target \
         /mission_manager/route /mission_manager/tile_markers /mission_manager/markers /mission_manager/terrain_markers \
         /exploration_complete /curtain/points /curtain/markers /tile_report; do
    printf '%s:\n  reliability: reliable\n  durability: transient_local\n  history: keep_last\n  depth: 100\n' "$t" >> "$QOS_FILE"
done

mkdir -p "$OUT_DIR"
# ros2 bag record refuses an existing folder: never overwrite an earlier bag, take the next free name instead
if [[ -e "$OUT_DIR/$NAME" ]]; then
    n=2
    while [[ -e "$OUT_DIR/${NAME}_$n" ]]; do n=$((n + 1)); done
    echo "$OUT_DIR/$NAME already exists: recording to ${NAME}_$n instead."
    NAME="${NAME}_$n"
fi
ARGS=(-o "$OUT_DIR/$NAME" -s "$STORAGE" --qos-profile-overrides-path "$QOS_FILE" --include-hidden-topics)
[[ "$MAX_SIZE_MB" != 0 ]] && ARGS+=(-b "$((MAX_SIZE_MB * 1024 * 1024))")

echo "Recording ${#TOPICS[@]} topics to $OUT_DIR/$NAME  (Ctrl-C to stop)"
exec ros2 bag record "${ARGS[@]}" "${TOPICS[@]}"
