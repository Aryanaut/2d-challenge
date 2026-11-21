#!/usr/bin/env bash
# Start the Lite3 tile explorer (lite3_explorer) in tmux, with live status panes and an RViz debug view.
#
#   window 1 "explorer": the full launch (SLAM, Nav2, mission manager, AI client)
#   window 2 "status"  : mission mode | mission events (decisions, stuck, recovery)
#                        tile report for every tile reached | chosen tile (what it is driving to)
#                        last AI answer | velocity sent to the dog / current gait
#   window 3 "health"  : /scan rate | /map rate | odom->base_link TF | rviz2 (Ctrl-C it to get a shell)
#   window 4 "bag"     : only with --bag: rosbag record (Ctrl-C there to stop it)
#
# RViz (scripts/explorer_debug.rviz) shows the tiles like the kitchen explorer: blue = visited (travelled to),
# green = to_visit (formed), red = blocked, purple = unreachable, orange = the CHOSEN tile (purple while in
# recovery) with its goal sphere, dark-blue trail, orange route (/mission_manager/route), cyan local plan, red goal arrow.
#
# Usage:
#   ./explorer.sh                    start (or re-attach to) the session
#   ./explorer.sh --scan             (alias --quickscan) run the quickscan animation (lean, look left/right) on every tile edge
#   ./explorer.sh --curtains         detect hanging curtains (depth camera, low rate); one along the current tile's edge is squared
#                                    up to and walked through with forward commands, then exploring goes on
#   ./explorer.sh --images           with --bag: also record the depth and (compressed) colour camera images, so the slats
#                                    detection can be replayed offline (large)
#   ./explorer.sh --bag [bagname]    also record a rosbag to bags/<bagname>_DD_MM_HHMMSS (default bagname: explorer)
#   ./explorer.sh stop               Ctrl-C the launch, stop the bag, close the session
#   Ctrl-b d / closing the session   same as stop; every process started from the session is killed
#
# Settings (environment variables):
#   SERVER_URL, GROUP_ID, LAUNCH_ARGS, WS (default ~/2d), ROS_DISTRO, SESSION (default explorer),
#   ROS_SETUP, ATTACH (0 = detached), RVIZ_CONFIG (default scripts/explorer_debug.rviz),
#   BAG_TOPICS  extra topics to record, space separated
#   CLOUD_TOPIC lidar pointcloud recorded in every bag (default /livox/points)
set -euo pipefail

SESSION="${SESSION:-explorer}"
WS="${WS:-$HOME/2d}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
SERVER_URL="${SERVER_URL:-}"
GROUP_ID="${GROUP_ID:-}"
LAUNCH_ARGS="${LAUNCH_ARGS:-}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/$ROS_DISTRO/setup.bash}"
ATTACH="${ATTACH:-1}"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
RVIZ_CONFIG="${RVIZ_CONFIG:-$HERE/explorer_debug.rviz}"
BAG_DIR="$WS/bags"
BAG_TOPICS="${BAG_TOPICS:-}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/livox/points}"     # raw lidar cloud, always recorded (terrain / floor analysis)

SOURCE="export EXPLORER_TAG=$SESSION; source $ROS_SETUP && source $WS/install/setup.bash"
LATCHED="--qos-durability transient_local --qos-reliability reliable"

# Everything needed to replay the exploration decisions: map, tiles, plans, goals, mode, commands, TF.
DEBUG_TOPICS="/tf /tf_static /map /scan /odom /cmd_vel /plan /local_plan /global_costmap/costmap /local_costmap/costmap /local_costmap/published_footprint /spin_crawl /spin_flat
/exploration_complete /terrain_type /terrain_classifier/details /terrain_classifier/slats_detections /terrain_kind /mission_manager/terrain_markers $CLOUD_TOPIC /tile_map /tile_report /llm/last_result
/mission_manager/mode /mission_manager/status /mission_manager/events /mission_manager/current_goal
/mission_manager/next_target /gait_switcher/gait /mission_manager/route /curtain/points /curtain/markers /virtual_walls /mission_manager/tile_markers /mission_manager/markers"

BAG=0
BAG_NAME=""
QUICKSCAN=0
IMAGES=0
CURTAINS=0
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)
            BAG=1
            if [[ $# -gt 1 && "$2" != -* && "$2" != stop ]]; then BAG_NAME="$2"; shift; fi ;;
        --quickscan|--scan) QUICKSCAN=1 ;;
        --images) IMAGES=1 ;;
        --curtains) CURTAINS=1 ;;
        --bag=*) BAG=1; BAG_NAME="${1#--bag=}" ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

command -v tmux >/dev/null || { echo "tmux is not installed: sudo apt install tmux" >&2; exit 1; }

# cleanup: kill every process started from this session's panes (they all carry EXPLORER_TAG=$SESSION in their
# environment, so children that survive the pane shell, e.g. nav2 / rviz2 / rosbag, are found too).
# SIGINT first (clean shutdown, bag is closed properly), then SIGTERM, then SIGKILL.
tagged_pids() {
    local f pid
    for f in /proc/[0-9]*/environ; do
        pid="${f#/proc/}"; pid="${pid%/environ}"
        [[ "$pid" == "$$" ]] && continue
        if tr '\0' '\n' < "$f" 2>/dev/null | grep -qx "EXPLORER_TAG=$SESSION"; then echo "$pid"; fi
    done
}
cleanup() {
    local sig pids i
    for sig in INT TERM KILL; do
        pids="$(tagged_pids)"
        [[ -z "$pids" ]] && return 0
        # shellcheck disable=SC2086
        kill -"$sig" $pids 2>/dev/null || true
        for i in $(seq 1 "$([[ $sig == KILL ]] && echo 2 || echo 16)"); do
            sleep 0.5
            [[ -z "$(tagged_pids)" ]] && return 0
        done
    done
}

if [[ "${1:-}" == "cleanup" ]]; then cleanup; exit 0; fi
# called by the tmux session-closed hook with the name of the closed session
if [[ "${1:-}" == "closed" ]]; then [[ "${2:-}" == "$SESSION" ]] && cleanup; exit 0; fi

if [[ "${1:-}" == "stop" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        if tmux list-windows -t "$SESSION" -F '#W' | grep -qx bag; then
            echo "Stopping the bag recording ..."
            tmux send-keys -t "$SESSION:bag" C-c
            sleep 3
        fi
        echo "Stopping launch (Ctrl-C) ..."
        tmux send-keys -t "$SESSION:explorer" C-c
        sleep 8          # let Nav2 cancel goals and nodes shut down cleanly
        tmux kill-session -t "$SESSION"
        cleanup          # anything still running (rviz2, stray nodes) is killed
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
[[ -f "$RVIZ_CONFIG" ]] || { echo "RViz config not found: $RVIZ_CONFIG" >&2; exit 1; }

LAUNCH="ros2 launch lite3_explorer exploration.launch.py"
[[ -n "$SERVER_URL" ]] && LAUNCH+=" server_url:=$SERVER_URL"
[[ -n "$GROUP_ID" ]] && LAUNCH+=" group_id:=$GROUP_ID"
[[ $QUICKSCAN == 1 ]] && LAUNCH+=" quickscan_on_edge:=true"
[[ $CURTAINS == 1 ]] && LAUNCH+=" use_curtains:=true"
[[ -n "$LAUNCH_ARGS" ]] && LAUNCH+=" $LAUNCH_ARGS"

[[ $IMAGES == 1 ]] && BAG_TOPICS="$BAG_TOPICS /camera/depth/image_raw /camera/color/image_raw/compressed"
if [[ $BAG == 1 ]]; then
    BAG_NAME="${BAG_NAME:-explorer}_$(date +%d_%m_%H%M%S)"     # name_DD_MM_HHMMSS
    BAG_NAME="${BAG_NAME//\//_}"          # a name, not a path: it always lands under bags/
    mkdir -p "$BAG_DIR"
    [[ -e "$BAG_DIR/$BAG_NAME" ]] && { echo "Bag already exists: $BAG_DIR/$BAG_NAME (pick another name)" >&2; exit 1; }
    echo "Recording rosbag to $BAG_DIR/$BAG_NAME"
fi

run() {
    tmux select-pane -t "$1" -T "$2"
    tmux send-keys -t "$1" "clear; printf '\\033[1;36m== %s ==\\033[0m\\n' '$2'; $SOURCE && $3" C-m
}

style_window() {
    tmux set-option -w -t "$1" pane-border-status top
    tmux set-option -w -t "$1" pane-border-format " #{pane_title} "
}

# ---- refuse to start on top of a stack that is still shutting down ------------------------------------
# A second launch next to a dying one gives two slam_toolbox / Nav2 stacks on the same TF tree (map->odom stops,
# "extrapolation into the past", "Invalid frame ID odom"). Nav2 can take ~50 s to die after Ctrl-C.
LEFT="$(bash -c "$SOURCE; ros2 node list 2>/dev/null" | grep -E '^/(slam_toolbox|planner_server|controller_server|bt_navigator|behavior_server|mission_manager|odom_tf_bridge)$' | sort -u | tr '\n' ' ' || true)"
if [[ -n "$LEFT" ]]; then
    echo "Nodes of a previous run are still alive: $LEFT" >&2
    echo "Wait for them to exit (or: pkill -INT -f nav2; ./explorer.sh stop), then start again." >&2
    exit 1
fi

# ---- Lite3 into auto mode, before anything is started ---------------------
# Same command as the lite3_auto alias in ~/.bashrc. Without it the robot ignores /cmd_vel.
LITE3_AUTO="timeout 15 ros2 topic pub /simple_cmd transfer_interfaces/msg/MotionSimpleCMD '{cmd_code: 0x21010C03, type: 0}' --once"
echo "Running lite3_auto ..."
bash -c "$SOURCE && $LITE3_AUTO" || { echo "lite3_auto failed; nothing started. Check the robot connection." >&2; exit 1; }

# ---- window 1: the launch -------------------------------------------------
tmux new-session -d -s "$SESSION" -n explorer -x 220 -y 60
tmux set-option -t "$SESSION" mouse on
# However the session ends (stop, Ctrl-b d, Ctrl-b &, last window closed), kill everything it started.
# Hooks and bindings are server-wide, so they only act on this session.
SELF="$(readlink -f "$0")"
tmux set-hook -g session-closed "run-shell -b \"SESSION=$SESSION '$SELF' closed '#{hook_session_name}'\""
tmux bind-key d if-shell -F "#{==:#{session_name},$SESSION}" \
    "run-shell -b \"SESSION=$SESSION '$SELF' stop\"" \
    "detach-client"
style_window "$SESSION:explorer"
run "$SESSION:explorer.0" "LAUNCH" "$LAUNCH"

# ---- window 2: exploration status ----------------------------------------
tmux new-window -t "$SESSION:" -n status
style_window "$SESSION:status"
tmux split-window -t "$SESSION:status.0" -v -l 75%
tmux split-window -t "$SESSION:status.1" -v -l 35%
tmux split-window -t "$SESSION:status.0" -h -l 50%
tmux split-window -t "$SESSION:status.3" -h -l 50%

run "$SESSION:status.0" "MISSION MODE (mode:phase)" \
    "ros2 topic echo /mission_manager/mode std_msgs/msg/String --field data $LATCHED"
run "$SESSION:status.1" "MISSION EVENTS" \
    "ros2 topic echo /mission_manager/events std_msgs/msg/String --field data"
run "$SESSION:status.2" "TILE REPORTS (what each tile looks like)" \
    "ros2 run lite3_explorer survey_monitor"
run "$SESSION:status.3" "CHOSEN TILE (driving to)" \
    "ros2 topic echo /mission_manager/next_target std_msgs/msg/String --field data $LATCHED | sed -u 's/^/DRIVING TO: /'"
run "$SESSION:status.4" "VELOCITY TO DOG (/cmd_vel)" \
    "ros2 topic echo /cmd_vel geometry_msgs/msg/Twist"

# current gait (latched /gait_switcher/gait): a small pane under the velocity pane
GAIT_PANE=$(tmux split-window -t "$SESSION:status.4" -v -l 30% -P -F '#{pane_id}')
run "$GAIT_PANE" "CURRENT GAIT (gait_switcher)" \
    "ros2 topic echo /gait_switcher/gait std_msgs/msg/String --field data $LATCHED"

# ---- window 3: health + rviz ----------------------------------------------
tmux new-window -t "$SESSION:" -n health
style_window "$SESSION:health"
tmux split-window -t "$SESSION:health.0" -h -l 50%
tmux split-window -t "$SESSION:health.0" -v -l 50%
tmux split-window -t "$SESSION:health.2" -v -l 50%
run "$SESSION:health.0" "/scan rate (lidar -> 2D)" "sleep 5; ros2 topic hz /scan"
run "$SESSION:health.1" "/map rate (slam_toolbox)" "sleep 5; ros2 topic hz /map"
run "$SESSION:health.2" "odom -> base_link TF" "sleep 5; ros2 run tf2_ros tf2_echo odom base_link"
run "$SESSION:health.3" "RVIZ2" \
    "echo 'Pause:  ros2 service call /mission_manager/set_active std_srvs/srv/SetBool \"{data: false}\"'; echo 'Resume: ros2 service call /mission_manager/set_active std_srvs/srv/SetBool \"{data: true}\"'; sleep 5; rviz2 -d '$RVIZ_CONFIG'"

# ---- window 4: rosbag (optional) -------------------------------------------
if [[ $BAG == 1 ]]; then
    tmux new-window -t "$SESSION:" -n bag
    style_window "$SESSION:bag"
    # shellcheck disable=SC2086
    run "$SESSION:bag.0" "ROSBAG -> bags/$BAG_NAME (Ctrl-C to stop)" \
        "sleep 5; ros2 bag record -o '$BAG_DIR/$BAG_NAME' $(echo $DEBUG_TOPICS $BAG_TOPICS)"
fi

tmux select-window -t "$SESSION:status"
if [[ "$ATTACH" == 1 ]]; then
    exec tmux attach -t "$SESSION"
fi
echo "Started detached. Attach with: tmux attach -t $SESSION"
