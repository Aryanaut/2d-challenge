#!/usr/bin/env bash
# Headless sim of the whole mission with simulated perception, then a check of every answer against the maze.
#   scripts/sim_mission_check.sh [maze.yaml] [seconds]       (default: lite3_sim mazes/mission_check.yaml, 300 s)
# Output: ~/lite3_sim_runs/mission_check_<time>/ (results.jsonl, scene_results.jsonl, *.log); exit code = the check's.
cd "$(dirname "$0")/.."
MAZE=$(realpath "${1:-src/lite3_sim/mazes/mission_check.yaml}")
DUR=${2:-300}
OUT=$HOME/lite3_sim_runs/mission_check_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}          # own domain: other sessions / the robot are not disturbed
echo "maze $MAZE, ${DUR}s, domain $ROS_DOMAIN_ID, output $OUT"

python3 -m lite3_sim.sim_perception --ros-args -p maze:="$MAZE" > "$OUT/perception.log" 2>&1 &
PERC=$!
python3 - "$OUT/scene_results.jsonl" > "$OUT/scene_recorder.log" 2>&1 <<'EOF' &
import sys, rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
rclpy.init(); n = rclpy.create_node('scene_result_recorder'); f = open(sys.argv[1], 'a')
qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(String, '/scene/result', lambda m: (f.write(m.data + '\n'), f.flush()), qos)
try:
    rclpy.spin(n)
except Exception:
    pass
EOF
REC=$!

timeout -s INT "$DUR" ros2 launch lite3_sim sim.launch.py maze:="$MAZE" rviz:=false reporter:=local \
    reporter_output_dir:="$OUT" > "$OUT/launch.log" 2>&1
kill -INT $PERC $REC 2>/dev/null
wait $PERC $REC 2>/dev/null
python3 scripts/sim_mission_check.py "$MAZE" "$OUT"
