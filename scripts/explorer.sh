#!/usr/bin/env bash
# Start the Lite3 tile explorer (lite3_explorer) in tmux, with live status panes and an RViz debug view.
#
#   window 1 "explorer": the full launch (SLAM, Nav2, mission manager, AI client)
#   window 2 "status"  : mission mode / controller driving now | mission events (decisions, stuck, recovery)
#                        tile report for every tile reached | chosen tile (what it is driving to)
#                        last AI answer | velocity sent to the dog / current gait / robot state (robot_state_bridge)
#   window 3 "health"  : /scan rate | /map rate | odom->base_link TF | rviz2 (Ctrl-C it to get a shell)
#   window 4 "bag"     : only with --bag: rosbag record (Ctrl-C there to stop it)
#   window "perception": only with --reporter: object/hazard/survivor detector + qr_scanner (perception.launch.py)
#   window "reporter"  : only with --reporter: the mission reporter (lite3_mission_reporter), POSTing to the chosen server URL
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
#   ./explorer.sh --stairs           detect staircases (depth camera, low rate); one on a tile beside the current tile is climbed
#                                    over in the robot's AI mode into the tile beyond, then walled off, then exploring goes on
#   ./explorer.sh --no-opening-stairs  do not run the stair detector during the opening scan (by default it does, and a
#                                    staircase beside the start tile is climbed right after the scan, even without --stairs)
#   ./explorer.sh --roller           detect roller tiles (depth camera, confirmed by the 3D lidar; low rate): the lidar maps the
#                                    roller as a wall across half the tile; one beside the current tile is squared up to and
#                                    crossed in the h-step gait into the tile beyond, the roller tile is marked explored
#   ./explorer.sh --roller-angle DEG  roller approach angle (default: mission.yaml modes.roller.approach_angle_deg, 20): the body stays
#                                    this far off the roller plane (+ = left, - = right, 0 = square on) while walking the straight line
#   ./explorer.sh --reporter URLA    also start perception + QR detection and the mission reporter, sending results to http://192.168.10.141:8001/task_report_A
#   ./explorer.sh --reporter URLB    ... to http://192.168.10.141:8001/task_report_B   (A / B and --reporter=URLA work too)
#   ./explorer.sh --images           with --bag: also record the depth and (compressed) colour camera images + camera_info, so the
#                                    slats / curtain detection can be replayed offline (large; implied by --curtains)
#   ./explorer.sh --bag [bagname]    also record a rosbag to bags/<bagname>_DD_MM_HHMMSS (default bagname: explorer);
#                                    it always includes the mission reporter's topics + /rosout (filled when --reporter runs)
#   ./explorer.sh --no-relay         no sensor_relay: every node (and the bag) subscribes to the robot topics itself (old behaviour)
#   ./explorer.sh stop               Ctrl-C the launch, stop the bag, close the session
#
# Robot sensor topics (/imu/data, /leg_odom2, /livox/points, the cameras) all come from one robot process that sends one
# copy per subscriber in a single loop: every extra subscriber to the cloud or an image used to drop the IMU from 200 Hz
# to 1-3 Hz. sensor_relay (in the launch) is the only subscriber and republishes them as /local/<topic>; the launch, the
# perception window and the bag all use those. So the bag records /local/imu/data etc.; replay under the robot names with
#   ros2 bag play <bag> --remap /local/imu/data:=/imu/data /local/leg_odom2:=/leg_odom2 /local/livox/points:=/livox/points ...
# For a rate check use `ros2 topic hz /local/imu/data` (or the relay's own log), never the robot topic.
#   Ctrl-b d / closing the session   same as stop; every process started from the session is killed
#
# Settings (environment variables):
#   SERVER_URL, GROUP_ID, LAUNCH_ARGS, WS (default ~/2d), ROS_DISTRO, SESSION (default explorer),
#   ROS_SETUP, ATTACH (0 = detached), RVIZ_CONFIG (default scripts/explorer_debug.rviz),
#   BAG_TOPICS  extra topics to record, space separated
#   CLOUD_TOPIC lidar pointcloud recorded in every bag (default /livox/points)
#   STATE_PORT  UDP port the 0x0901 robot state arrives on (default 43897)
#   STATE_IP    local address robot_state_bridge listens on (default 192.168.1.71, where relay_robot_state.py on
#               192.168.1.50 sends the reports; 0.0.0.0 = all interfaces)
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
RELAY=1
SP=/local       # sensor prefix: robot sensor topics are read as $SP<topic> (sensor_relay); '' with --no-relay

SOURCE="export EXPLORER_TAG=$SESSION; source $ROS_SETUP && source $WS/install/setup.bash"
LATCHED="--qos-durability transient_local --qos-reliability reliable"

# Everything needed to replay the exploration decisions: map, tiles, plans, goals, mode, commands, TF.
DEBUG_TOPICS="/tf /tf_static /map /scan /odom /cmd_vel /plan /local_plan /global_costmap/costmap /local_costmap/costmap /local_costmap/published_footprint /spin_crawl /spin_flat
/exploration_complete /terrain_type /terrain_classifier/details /terrain_classifier/slats_detections /terrain_kind /mission_manager/terrain_markers __SP__$CLOUD_TOPIC /tile_map /tile_report /llm/last_result
/mission_manager/mode /mission_manager/status /mission_manager/events /mission_manager/current_goal
/mission_manager/next_target /gait_switcher/gait /mission_manager/route /curtain/points /curtain/markers /curtain/wall /stairs/points /stairs/markers /stairs/enable /roller/points /roller/markers /roller_mode /step_up /ai_mode /simple_cmd /controller_monitor/mppi_omni /controller_monitor/active /virtual_walls /mission_manager/tile_markers /mission_manager/markers /robot_state /robot_state/text __SP__/imu/data __SP__/leg_odom2 /odom_rebase /scene/looking"

BAG=0
BAG_NAME=""
QUICKSCAN=0
IMAGES=0
CURTAINS=0
STAIRS=0
OPENING_STAIRS=1
ROLLER=0
ROLLER_ANGLE=
REPORTER=
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)
            BAG=1
            if [[ $# -gt 1 && "$2" != -* && "$2" != stop ]]; then BAG_NAME="$2"; shift; fi ;;
        --quickscan|--scan) QUICKSCAN=1 ;;
        --images) IMAGES=1 ;;
        --curtains) CURTAINS=1 ;;
        --stairs) STAIRS=1 ;;
        --no-opening-stairs) OPENING_STAIRS=0 ;;
        --roller|--rollers) ROLLER=1 ;;
        --roller-angle) ROLLER=1; ROLLER_ANGLE="${2:?--roller-angle needs a number of degrees}"; shift ;;
        --roller-angle=*) ROLLER=1; ROLLER_ANGLE="${1#--roller-angle=}" ;;
        --reporter) REPORTER="${2:?--reporter needs URLA or URLB}"; shift ;;
        --reporter=*) REPORTER="${1#--reporter=}" ;;
        --bag=*) BAG=1; BAG_NAME="${1#--bag=}" ;;
        --no-relay) RELAY=0; SP= ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

if [[ -n $REPORTER ]]; then
    REPORTER="${REPORTER^^}"; REPORTER="${REPORTER#URL}"
    [[ "$REPORTER" == A || "$REPORTER" == B ]] || { echo "--reporter takes URLA or URLB (got '$REPORTER')" >&2; exit 1; }
fi

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
[[ $STAIRS == 1 ]] && LAUNCH+=" use_stairs:=true"
[[ $OPENING_STAIRS == 0 ]] && LAUNCH+=" stairs_on_opening:=false"
[[ $ROLLER == 1 ]] && LAUNCH+=" use_rollers:=true"
[[ -n $ROLLER_ANGLE ]] && LAUNCH+=" roller_angle_deg:=$ROLLER_ANGLE"
[[ $RELAY == 0 ]] && LAUNCH+=" use_relay:=false"
[[ -n "$LAUNCH_ARGS" ]] && LAUNCH+=" $LAUNCH_ARGS"

[[ $CURTAINS == 1 ]] && IMAGES=1     # the curtain detector (depth + colour) can only be replayed with the images
# every bag records the mission reporter: its output, all of its inputs, and /rosout for its log lines (held / pending /
# POSTed); small topics, no images. Topics not published (no --reporter) are simply absent from the bag.
REPORTER_TOPICS="/mission/result /mission/manual_report /mission/scene_type /scene/result /robot/position /perception/event /perception/llm_result /perception/qr_text /qr/visible /qr/detections /perception/detections /rosout"
BAG_TOPICS="$BAG_TOPICS $REPORTER_TOPICS"
[[ $IMAGES == 1 ]] && BAG_TOPICS="$BAG_TOPICS $SP/camera/depth/image_raw $SP/camera/depth/camera_info $SP/camera/color/image_raw/compressed $SP/camera/color/camera_info"
DEBUG_TOPICS="${DEBUG_TOPICS//__SP__/$SP}"
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

# robot state as the robot reports it (UDP 0x0901 -> /robot_state, /robot_state/text). The motion host only reports to
# 192.168.1.50 (jetson2motion); relay_robot_state.py there forwards it to STATE_IP:STATE_PORT. robot_state_bridge logs one
# line per change, e.g. "robot state (6,13,1) stepping, RUG H-step  battery 87%", "(16,0,1) AI basic gait".
# the mission's stairs mode uses /robot_state to resend AI on / off until the robot reports it.
STATE_PANE=$(tmux split-window -t "$GAIT_PANE" -v -l 50% -P -F '#{pane_id}')
run "$STATE_PANE" "ROBOT STATE (basic,gait|policy,motion)" \
    "ros2 run lite3_explorer robot_state_bridge --ros-args -p bind_ip:=${STATE_IP:-192.168.1.71} -p port:=${STATE_PORT:-43897} || echo 'robot_state_bridge failed: not built (colcon build --packages-select lite3_explorer), or STATE_IP ${STATE_IP:-192.168.1.71} is not an address of this computer (STATE_IP=0.0.0.0 = all interfaces)'"

# controller in charge (Nav2 FollowPath RPP / MPPI, RecoveryMPPI, Spin, or the mission's own /cmd_vel): under the mode pane
CTRL_PANE=$(tmux split-window -t "$SESSION:status.0" -v -l 60% -P -F '#{pane_id}')
run "$CTRL_PANE" "CONTROLLER (who drives the dog now)" "ros2 run lite3_explorer controller_monitor"

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

# ---- window: mission reporter (optional) ----------------------------------
if [[ -n $REPORTER ]]; then
    # perception (object / hazard / survivor detector) + qr_scanner: the reporter's only source of results
    tmux new-window -t "$SESSION:" -n perception
    style_window "$SESSION:perception"
    run "$SESSION:perception.0" "PERCEPTION + QR SCANNER" \
        "ros2 launch lite3_perception perception.launch.py with_qr:=true sensor_prefix:=$SP"
    tmux new-window -t "$SESSION:" -n reporter
    style_window "$SESSION:reporter"
    run "$SESSION:reporter.0" "MISSION REPORTER (URL$REPORTER)" \
        "ros2 launch lite3_mission_reporter mission_reporter.launch.py server_url:=http://192.168.10.141:8001/task_report_$REPORTER"
    # live view of every result sent (/mission/result, same JSON as the POST body)
    RES_PANE=$(tmux split-window -t "$SESSION:reporter.0" -v -l 40% -P -F '#{pane_id}')
    run "$RES_PANE" "MISSION RESULT (what is sent to server $REPORTER)" \
        "ros2 topic echo /mission/result std_msgs/msg/String --field data $LATCHED"
fi

tmux select-window -t "$SESSION:status"
if [[ "$ATTACH" == 1 ]]; then
    exec tmux attach -t "$SESSION"
fi
echo "Started detached. Attach with: tmux attach -t $SESSION"
