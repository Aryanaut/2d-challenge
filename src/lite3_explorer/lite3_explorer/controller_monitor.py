"""Print which controller drives the dog right now (tmux status window of scripts/explorer.sh).

  ros2 run lite3_explorer controller_monitor

At start it asks Nav2's controller_server which plugins its FollowPath / RecoveryMPPI controllers are (that is what
controller:=rpp | mppi on exploration.launch.py chose); in lite3_sim, which has no controller_server, it reads the
simulator's controller.type instead. Then it follows /mission_manager/mode and /mission_manager/events and prints a
line whenever the one in charge changes:

  exploring / backtracking driving   Nav2 FollowPath (RegulatedPurePursuit or MPPI)
  recovery driving                   Nav2 RecoveryMPPI, or 'omni' (mission manager /cmd_vel) after a recovery_fallback
  scanning turning                   Nav2 behavior_server Spin
  curtain / stairs / roller / mppi / scene    mission manager, direct /cmd_vel
  anything else                      nothing is driving

It also reads each MPPI controller's motion_model, and publishes (latched):
  /controller_monitor/mppi_omni  std_msgs/Bool    true while an MPPI controller with motion_model Omni drives the dog
                                                  (controller:=rpp: only recovery's RecoveryMPPI; controller:=mppi: every
                                                  Nav2 path, FollowPath too; never Spin or the mission's own /cmd_vel)
  /controller_monitor/active     std_msgs/String  the line printed last, e.g. 'Nav2 FollowPath = MPPI (Omni)'
In lite3_sim the one simulated controller (controller.type) drives everything: omni stands in for MPPI Omni.
"""
import datetime as dt
import json

import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

DIRECT = ('curtain', 'stairs', 'roller', 'mppi', 'scene')          # modes that publish /cmd_vel themselves
NAV2_DRIVING = ('driving', 'to_split', 'to_curtain', 'to_roller', 'to_option')


def short(plugin):
    """'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController' -> 'RegulatedPurePursuit'."""
    name = (plugin or '?').split('::')[-1].split('/')[-1]
    return name[:-len('Controller')] if name.endswith('Controller') and name != 'Controller' else name


class ControllerMonitor(Node):
    def __init__(self):
        super().__init__('controller_monitor')
        self.plugins = {}                                  # controller id -> {'label', 'omni_mppi'}
        self.source = None                                 # where self.plugins came from
        self.mode, self.phase, self.fallback = '', '', False
        self.last = None
        self.param_clients = {'nav2': self.create_client(GetParameters, '/controller_server/get_parameters'),
                        'sim': self.create_client(GetParameters, '/lite3_sim/get_parameters')}
        self._pending = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/mission_manager/mode', self._on_mode, latched)
        self.create_subscription(String, '/mission_manager/events', self._on_event, 50)
        self.omni_pub = self.create_publisher(Bool, '/controller_monitor/mppi_omni', latched)
        self.active_pub = self.create_publisher(String, '/controller_monitor/active', latched)
        self.omni = None
        self.create_timer(1.0, self._ask)
        self._ask()

    # -- what the controllers are
    def _ask(self):
        if self.source is not None or self._pending is not None:
            return
        if self.param_clients['nav2'].service_is_ready():
            names = ['FollowPath.plugin', 'FollowPath.motion_model', 'RecoveryMPPI.plugin', 'RecoveryMPPI.motion_model']
            fut = self.param_clients['nav2'].call_async(GetParameters.Request(names=names))
            fut.add_done_callback(lambda f: self._on_params('Nav2 controller_server', names, f))
        elif self.param_clients['sim'].service_is_ready():
            names = ['controller.type']
            fut = self.param_clients['sim'].call_async(GetParameters.Request(names=names))
            fut.add_done_callback(lambda f: self._on_params('lite3_sim', names, f))
        else:
            self._say('waiting for Nav2 controller_server (or lite3_sim) ...', key='waiting')
            return
        self._pending = fut

    def _on_params(self, source, names, fut):
        self._pending = None
        try:
            values = [v.string_value for v in fut.result().values]
        except Exception as exc:  # noqa: BLE001 - not up yet: try again next second
            self.get_logger().warning(f'{source}: {exc}')
            return
        if source == 'lite3_sim':
            # one simulated controller for every Nav2 goal (recovery too): omni stands in for MPPI Omni
            kind = values[0] or 'rpp'
            omni = kind == 'omni'
            entry = {'label': 'MPPI (Omni, simulated)' if omni else f'{kind.upper()} (simulated)', 'omni_mppi': omni}
            self.plugins = {'FollowPath': entry, 'RecoveryMPPI': dict(entry)}
        else:
            got = dict(zip(names, values))
            self.plugins = {}
            for cid in ('FollowPath', 'RecoveryMPPI'):
                plugin = got.get(f'{cid}.plugin')
                if not plugin:
                    continue
                name, model = short(plugin), got.get(f'{cid}.motion_model') or ''
                mppi = 'MPPI' in name
                self.plugins[cid] = {'label': f'{name} ({model or "DiffDrive"})' if mppi else name,
                                     'omni_mppi': mppi and model.lower() == 'omni'}
        self.source = source
        self._say('configured: ' + ', '.join(f'{k} = {v["label"]}' for k, v in self.plugins.items())
                  + f'  (from {source})',
                  key='configured')
        self._show()

    # -- who is driving now
    def _on_mode(self, msg):
        mode, _, phase = msg.data.partition(':')
        if mode != self.mode:
            self.fallback = False                          # a new mode: no recovery fallback yet
        self.mode, self.phase = mode, phase
        self._show()

    def _on_event(self, msg):
        try:
            kind = json.loads(msg.data).get('kind')
        except ValueError:
            return
        if kind == 'recovery_fallback':
            self.fallback = True
            self._show()

    def _label(self, cid):
        return self.plugins.get(cid, {}).get('label', '?')

    def _now_driving(self):
        """(what drives now, the Nav2 controller id driving or None)."""
        if self.mode in DIRECT:
            return f'mission manager /cmd_vel ({self.mode} mode)', None
        if self.mode == 'scanning' and self.phase == 'turning':
            return 'Nav2 behavior_server Spin', None
        if self.mode == 'recovery' and self.phase == 'driving':
            if self.fallback:
                return 'omni fallback: mission manager /cmd_vel (not MPPI)', None
            return f'Nav2 RecoveryMPPI = {self._label("RecoveryMPPI")}', 'RecoveryMPPI'
        if self.phase in NAV2_DRIVING:
            return f'Nav2 FollowPath = {self._label("FollowPath")}', 'FollowPath'
        return f'none (Nav2 FollowPath = {self._label("FollowPath")} when driving)', None

    def _show(self):
        text, cid = self._now_driving()
        omni = bool(cid and self.plugins.get(cid, {}).get('omni_mppi'))
        state = f'{self.mode}:{self.phase}' if self.phase else (self.mode or 'no mission mode yet')
        if omni != self.omni:
            self.omni = omni
            self.omni_pub.publish(Bool(data=omni))
        if text != getattr(self, '_active', None):
            self._active = text
            self.active_pub.publish(String(data=text))
        self._say(f'{"MPPI OMNI >> " if omni else "             "}{text:55s} [{state}]', key=text)

    def _say(self, text, key):
        if key == self.last:
            return
        self.last = key
        print(f'{dt.datetime.now():%H:%M:%S}  {text}', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
