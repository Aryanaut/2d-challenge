#!/usr/bin/env bash
# Start the Lite3 kitchen explorer in tmux, with live status panes.
#
#   window 1 "explorer": the full launch (SLAM, Nav2, frontier explorer, AI mission manager)
#   window 2 "status"  : mission state | explorer status
#                        last AI result (full width)
#                        current navigation goal (what it is driving to, tile or frontier)
#                        kitchen found / exploration complete | velocity sent to the dog | walking mode
#   window 3 "health"  : /scan rate | /map rate | odom->base_link TF | rviz2 (Ctrl-C it to get a shell)
#
# Usage:
#   ./run_kitchen_tmux.sh                 start (or re-attach to) the session
#   ./run_kitchen_tmux.sh --no-ai         start the kitchen explorer only: the lite3_llm_tools llm_client is NOT started
#                                         (same as USE_AI=false)
#   ./run_kitchen_tmux.sh --ai            kitchen explorer + llm_client (same as USE_AI=true; this is the default)
#   ./run_kitchen_tmux.sh stop            Ctrl-C the launch, close the session, then run lite3_manual
#   Ctrl-b d (inside the session)         same as stop; use Ctrl-b : detach-client to only detach
#
# Lite3 mode: start runs lite3_auto before the launch; stop runs lite3_manual after the session is closed.
#
# Settings (environment variables):
#   SERVER_URL   AI server, e.g. http://192.168.10.141:8000  (empty = use lite3_llm_tools config/llm_client.yaml)
#   GROUP_ID     your group id                              (empty = use lite3_llm_tools config/llm_client.yaml)
#   USE_AI       true/false (default true): true runs the kitchen explorer and the lite3_llm_tools llm_client;
#                false runs only the kitchen explorer: no llm_client, no camera views / spins, tiles placed from the
#                walls and the nearest connected tile chosen.
#   USE_TERRAIN  true/false: run the rugged / non-rugged classifier (default true)
#   LAUNCH_ARGS  extra launch arguments, e.g. "publish_lidar_tf:=true lidar_x:=0.15 lidar_z:=0.12"
#   WS           ROS 2 workspace          (default ~/2d)
#   ROS_DISTRO   ROS 2 distribution       (default humble)
#   SESSION      tmux session name        (default kitchen)
#   ROS_SETUP    ROS setup file           (default /opt/ros/$ROS_DISTRO/setup.bash)
#   ATTACH       0 = start detached       (default 1)
#   RVIZ_CONFIG  .rviz file for rviz2     (empty = rviz2 default view)
#
# Example:
#   SERVER_URL=http://192.168.10.141:8000 GROUP_ID=team7 ./run_kitchen_tmux.sh
set -euo pipefail

SESSION="${SESSION:-kitchen}"
WS="${WS:-$HOME/2d}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
SERVER_URL="${SERVER_URL:-}"
GROUP_ID="${GROUP_ID:-}"
USE_AI="${USE_AI:-true}"
USE_TERRAIN="${USE_TERRAIN:-true}"
LAUNCH_ARGS="${LAUNCH_ARGS:-}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/$ROS_DISTRO/setup.bash}"
ATTACH="${ATTACH:-1}"
RVIZ_CONFIG="${RVIZ_CONFIG:-}"

# Defined before the "stop" branch, which needs them (set -u aborts on unset variables).
SOURCE="source $ROS_SETUP && source $WS/install/setup.bash"
LATCHED="--qos-durability transient_local --qos-reliability reliable"

# Lite3 mode commands (same as the lite3_auto / lite3_manual aliases in ~/.bashrc)
LITE3_AUTO="timeout 15 ros2 topic pub /simple_cmd transfer_interfaces/msg/MotionSimpleCMD '{cmd_code: 0x21010C03, type: 0}' --once"
LITE3_MANUAL="timeout 15 ros2 topic pub /simple_cmd transfer_interfaces/msg/MotionSimpleCMD '{cmd_code: 0x21010C02, type: 0}' --once"

# --ai / --no-ai override USE_AI; any other argument (e.g. "stop") is left for the command below.
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --ai) USE_AI=true ;;
        --no-ai) USE_AI=false ;;
        *) ARGS+=("$arg") ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
case "${USE_AI,,}" in
    ""|true|false) ;;
    *) echo "USE_AI must be true or false (got '$USE_AI')" >&2; exit 1 ;;
esac

command -v tmux >/dev/null || { echo "tmux is not installed: sudo apt install tmux" >&2; exit 1; }

if [[ "${1:-}" == "stop" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Stopping launch (Ctrl-C) ..."
        tmux send-keys -t "$SESSION:explorer" C-c
        sleep 8          # let Nav2 cancel goals and nodes shut down cleanly
        tmux kill-session -t "$SESSION"
        echo "Session '$SESSION' closed."
        echo "Running lite3_manual ..."
        bash -c "$SOURCE && $LITE3_MANUAL" || echo "lite3_manual failed; check the robot state manually." >&2
    else
        echo "No tmux session '$SESSION'."
    fi
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running; attaching."
    [[ "$ATTACH" == 1 ]] && exec tmux attach -t "$SESSION"
    exit 0
fi

[[ -f "$ROS_SETUP" ]] || { echo "ROS 2 setup not found: $ROS_SETUP" >&2; exit 1; }
[[ -f "$WS/install/setup.bash" ]] || { echo "No $WS/install/setup.bash; build the workspace first" >&2; exit 1; }

LAUNCH="ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py"
[[ -n "$SERVER_URL" ]] && LAUNCH+=" server_url:=$SERVER_URL"
[[ -n "$GROUP_ID" ]] && LAUNCH+=" group_id:=$GROUP_ID"
LAUNCH+=" use_ai:=${USE_AI,,}"
LAUNCH+=" use_terrain:=$USE_TERRAIN"
[[ -n "$LAUNCH_ARGS" ]] && LAUNCH+=" $LAUNCH_ARGS"
echo "Tile report (what each tile looks like): ~/2d/reports/tile_report_<run>.md"
echo "llm_client (lite3_llm_tools): $([[ "${USE_AI,,}" == true ]] && echo "ON, started with the kitchen explorer" || echo "OFF (--no-ai), not started")"

# run <pane> <title> <command>: label the pane, source ROS, run the command.
# Commands are typed into a shell, so the pane stays open if the command exits.
# The title is also printed, because some shell prompts overwrite pane titles.
run() {
    tmux select-pane -t "$1" -T "$2"
    tmux send-keys -t "$1" "clear; printf '\\033[1;36m== %s ==\\033[0m\\n' '$2'; $SOURCE && $3" C-m
}

style_window() {
    tmux set-option -w -t "$1" pane-border-status top
    tmux set-option -w -t "$1" pane-border-format " #{pane_title} "
}

# ---- window 1: the launch -------------------------------------------------
tmux new-session -d -s "$SESSION" -n explorer -x 220 -y 60
tmux set-option -t "$SESSION" mouse on
# Ctrl-b d in this session runs "stop" (Ctrl-C, close session, lite3_manual).
# Bindings are server-wide, so other sessions still just detach.
SELF="$(readlink -f "$0")"
tmux bind-key d if-shell -F "#{==:#{session_name},$SESSION}" \
    "run-shell -b \"SESSION=$SESSION '$SELF' stop\"" \
    "detach-client"
style_window "$SESSION:explorer"
# lite3_auto runs first; the launch only starts if it succeeds.
run "$SESSION:explorer.0" "LAUNCH" "$LITE3_AUTO && $LAUNCH"

# ---- window 2: exploration status ----------------------------------------
tmux new-window -t "$SESSION" -n status
style_window "$SESSION:status"
tmux split-window -t "$SESSION:status.0" -v -l 75%   # 0 top | 1 rest
tmux split-window -t "$SESSION:status.1" -v -l 55%   # 1 middle | 2 bottom
tmux split-window -t "$SESSION:status.2" -v -l 50%   # 2 middle | 3 bottom
tmux split-window -t "$SESSION:status.0" -h -l 50%   # top: 0 left | 1 right  (rest renumber to 2, 3, 4)
tmux split-window -t "$SESSION:status.4" -h -l 50%   # bottom: 4 left | 5 right
tmux split-window -t "$SESSION:status.5" -h -l 35% -d   # bottom right: 5 velocity | 6 walking mode

run "$SESSION:status.0" "MISSION STATE" \
    "ros2 topic echo /kitchen_mission_manager/state std_msgs/msg/String --field data $LATCHED"
run "$SESSION:status.1" "EXPLORER STATUS" \
    "ros2 topic echo /frontier_explorer/status std_msgs/msg/String --field data"
run "$SESSION:status.2" "LAST AI RESULT" \
    "ros2 topic echo /llm/last_result std_msgs/msg/String --field data"
run "$SESSION:status.3" "CURRENT NAV GOAL" \
    "(ros2 topic echo /environment_reporter/next_target std_msgs/msg/String --field data $LATCHED | sed -u 's/^/AI picked: /' &) ; ros2 topic echo /frontier_explorer/goal_info std_msgs/msg/String --field data $LATCHED | sed -u 's/^/DRIVING TO: /'"
run "$SESSION:status.4" "KITCHEN FOUND / EXPLORATION COMPLETE" \
    "(ros2 topic echo /exploration_complete std_msgs/msg/Bool --field data $LATCHED | sed -u 's/^/complete: /' &) ; ros2 topic echo /kitchen_found geometry_msgs/msg/PoseStamped $LATCHED"
run "$SESSION:status.5" "VELOCITY TO DOG (/cmd_vel)" \
    "ros2 topic echo /cmd_vel geometry_msgs/msg/Twist"
# Walking mode, from /terrain_type (latched): RUGGED or NON-RUGGED
run "$SESSION:status.6" "WALKING MODE" \
    "ros2 topic echo /terrain_type std_msgs/msg/String --field data $LATCHED | sed -u 's/^non_rugged\$/NON-RUGGED (flat gait)/; s/^rugged\$/RUGGED (rugged gait)/; s/^unknown\$/UNKNOWN (no terrain data yet)/'"

# ---- window 3: health ------------------------------------------------------
tmux new-window -t "$SESSION" -n health
style_window "$SESSION:health"
tmux split-window -t "$SESSION:health.0" -h -l 50%
tmux split-window -t "$SESSION:health.0" -v -l 50%
tmux split-window -t "$SESSION:health.2" -v -l 50%
run "$SESSION:health.0" "/scan rate (lidar -> 2D)" "sleep 5; ros2 topic hz /scan"
run "$SESSION:health.1" "/map rate (slam_toolbox)" "sleep 5; ros2 topic hz /map"
run "$SESSION:health.2" "odom -> base_link TF" "sleep 5; ros2 run tf2_ros tf2_echo odom base_link"
RVIZ_CMD="rviz2"
[[ -n "$RVIZ_CONFIG" ]] && RVIZ_CMD+=" -d $RVIZ_CONFIG"
run "$SESSION:health.3" "RVIZ2" \
    "echo 'Pause:  ros2 service call /frontier_explorer/set_active std_srvs/srv/SetBool \"{data: false}\"'; echo 'Resume: ros2 service call /frontier_explorer/set_active std_srvs/srv/SetBool \"{data: true}\"'; echo 'Query images + answers: ~/lite3_kitchen_runs/'; sleep 5; $RVIZ_CMD"

tmux select-window -t "$SESSION:status"
if [[ "$ATTACH" == 1 ]]; then
    exec tmux attach -t "$SESSION"
fi
echo "Started detached. Attach with: tmux attach -t $SESSION"