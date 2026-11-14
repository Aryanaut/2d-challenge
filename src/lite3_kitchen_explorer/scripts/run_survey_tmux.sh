#!/usr/bin/env bash
# Start the Lite3 frontier explorer + AI environment reporter in tmux, with live status panes.
#
#   window 1 "explorer": the full launch (SLAM, Nav2, frontier explorer, AI environment reporter)
#   window 2 "status"  : reporter state | explorer status
#                        environment report for every stop (full width)
#                        last AI answer per view | velocity sent to the dog
#   window 3 "health"  : /scan rate | /map rate | odom->base_link TF | free shell
#
# Usage:
#   ./run_survey_tmux.sh                 start (or re-attach to) the session
#   ./run_survey_tmux.sh stop            Ctrl-C the launch, then close the session
#
# Settings (environment variables):
#   SERVER_URL   AI server, e.g. http://192.168.10.141:8000  (empty = use lite3_llm_tools config/llm_client.yaml)
#   GROUP_ID     your group id                              (empty = use lite3_llm_tools config/llm_client.yaml)
#   LAUNCH_ARGS  extra launch arguments, e.g. "publish_lidar_tf:=true lidar_x:=0.15 lidar_z:=0.12"
#   WS           ROS 2 workspace          (default ~/ros2_ws)
#   ROS_DISTRO   ROS 2 distribution       (default humble)
#   SESSION      tmux session name        (default survey)
#   ROS_SETUP    ROS setup file           (default /opt/ros/$ROS_DISTRO/setup.bash)
#   ATTACH       0 = start detached       (default 1)
#
# Example:
#   SERVER_URL=http://192.168.10.141:8000 GROUP_ID=team7 ./run_survey_tmux.sh
set -euo pipefail

SESSION="${SESSION:-survey}"
WS="${WS:-$HOME/ros2_ws}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
SERVER_URL="${SERVER_URL:-}"
GROUP_ID="${GROUP_ID:-}"
LAUNCH_ARGS="${LAUNCH_ARGS:-}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/$ROS_DISTRO/setup.bash}"
ATTACH="${ATTACH:-1}"

command -v tmux >/dev/null || { echo "tmux is not installed: sudo apt install tmux" >&2; exit 1; }

if [[ "${1:-}" == "stop" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Stopping launch (Ctrl-C) ..."
        tmux send-keys -t "$SESSION:explorer" C-c
        sleep 8          # let Nav2 cancel goals and nodes shut down cleanly
        tmux kill-session -t "$SESSION"
        echo "Session '$SESSION' closed."
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

SOURCE="source $ROS_SETUP && source $WS/install/setup.bash"
LATCHED="--qos-durability transient_local --qos-reliability reliable"

LAUNCH="ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py"
[[ -n "$SERVER_URL" ]] && LAUNCH+=" server_url:=$SERVER_URL"
[[ -n "$GROUP_ID" ]] && LAUNCH+=" group_id:=$GROUP_ID"
[[ -n "$LAUNCH_ARGS" ]] && LAUNCH+=" $LAUNCH_ARGS"

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
style_window "$SESSION:explorer"
run "$SESSION:explorer.0" "LAUNCH" "$LAUNCH"

# ---- window 2: exploration status ----------------------------------------
tmux new-window -t "$SESSION" -n status
style_window "$SESSION:status"
tmux split-window -t "$SESSION:status.0" -v -l 75%   # 0 top | 1 rest
tmux split-window -t "$SESSION:status.1" -v -l 35%   # 1 middle | 2 bottom
tmux split-window -t "$SESSION:status.0" -h -l 50%   # top: 0 left | 1 right  (rest renumber to 2, 3)
tmux split-window -t "$SESSION:status.3" -h -l 50%   # bottom: 3 left | 4 right

run "$SESSION:status.0" "REPORTER STATE" \
    "ros2 topic echo /environment_reporter/state std_msgs/msg/String --field data $LATCHED"
run "$SESSION:status.1" "EXPLORER STATUS" \
    "ros2 topic echo /frontier_explorer/status std_msgs/msg/String --field data"
run "$SESSION:status.2" "TILE REPORTS (what each tile looks like)" \
    "ros2 run lite3_kitchen_explorer survey_monitor"
run "$SESSION:status.3" "LAST AI ANSWER" \
    "ros2 topic echo /llm/last_result std_msgs/msg/String --field data"
run "$SESSION:status.4" "VELOCITY TO DOG (/cmd_vel)" \
    "ros2 topic echo /cmd_vel geometry_msgs/msg/Twist"

# ---- window 3: health ------------------------------------------------------
tmux new-window -t "$SESSION" -n health
style_window "$SESSION:health"
tmux split-window -t "$SESSION:health.0" -h -l 50%
tmux split-window -t "$SESSION:health.0" -v -l 50%
tmux split-window -t "$SESSION:health.2" -v -l 50%
run "$SESSION:health.0" "/scan rate (lidar -> 2D)" "sleep 5; ros2 topic hz /scan"
run "$SESSION:health.1" "/map rate (slam_toolbox)" "sleep 5; ros2 topic hz /map"
run "$SESSION:health.2" "odom -> base_link TF" "sleep 5; ros2 run tf2_ros tf2_echo odom base_link"
run "$SESSION:health.3" "SHELL" \
    "echo 'Pause:  ros2 service call /frontier_explorer/set_active std_srvs/srv/SetBool \"{data: false}\"'; echo 'Resume: ros2 service call /frontier_explorer/set_active std_srvs/srv/SetBool \"{data: true}\"'; echo 'Describe this tile now: ros2 service call /environment_reporter/report_now std_srvs/srv/Trigger'; echo 'Tile report: ~/2d/reports/tile_report_<run>.md   images: ~/lite3_survey_runs/<run>/'"

tmux select-window -t "$SESSION:status"
if [[ "$ATTACH" == 1 ]]; then
    exec tmux attach -t "$SESSION"
fi
echo "Started detached. Attach with: tmux attach -t $SESSION"
