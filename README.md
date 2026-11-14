# 2d-challenge
Robot dawg

## Recording a debug rosbag

This is the command used for the `bags/route_debug_*` recordings (run it while the kitchen exploration stack is up, from the
workspace root). It captures the AI's tile choice, the route waypoints sent for it, the goal the explorer actually
drives to, the Nav2 action status/feedback, the costmaps and the robot state, so you can see where the route and the
chosen tile diverge.

```bash
cd ~/2d && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 bag record \
  -o bags/large_route_debug_$(date +%H%M%S) \
  --include-hidden-topics \
  /tf /tf_static /odom /camera/color/image_raw/compressed /map /cmd_vel \
  /plan /received_global_plan /local_plan \
  /tile_report \
  /environment_reporter/state \
  /llm/request /llm/response /llm/status /llm/last_result \
  /environment_reporter/next_target /environment_reporter/markers \
  /environment_reporter/tile_markers \
  /tile_map \
  /frontier_explorer/preferred_goal /frontier_explorer/preferred_route \
  /frontier_explorer/current_goal /frontier_explorer/goal_info /frontier_explorer/goal_failed \
  /frontier_explorer/status /frontier_explorer/markers \
  /frontier_list /frontier_candidates /frontier_reached /exploration_complete \
  /navigate_to_pose/_action/status /navigate_to_pose/_action/feedback \
  /follow_waypoints/_action/status /follow_waypoints/_action/feedback \
  /compute_path_to_pose/_action/status \
  /global_costmap/costmap /global_costmap/costmap_updates \
  /local_costmap/costmap \
  /spin_crawl /simple_cmd /terrain_type \
  /rosout
```

Notes:
- Default storage (sqlite3, no compression).
- `/local_plan`, `/exploration_complete` and `/global_costmap/costmap_updates` were in the command but had no messages
  in the first recordings (nothing published them during the run). The `/llm/*` topics only exist once the
  `lite3_llm_tools` `llm_client` node runs; they carry every AI request, its answer and the client's status. Each `/llm/request` has the same
  `id` as its `/llm/response`.
- The bag directory name is the start time (`HHMMSS`); change the `-o` prefix for a different run.
- Inspect it with `ros2 bag info bags/<name>`; the AI choice and route topics are `/environment_reporter/next_target`,
  `/frontier_explorer/preferred_route` and `/frontier_explorer/goal_info`.
