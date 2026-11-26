"""Lite3 robot state report (motion host -> perception host, UDP) and what each gait / AI command should lead to.

DEEP Robotics Motion Host Communication Interface, sec 1.3.1 Robot State: command code 0x0901, type 1, 50 Hz,
payload struct RobotStateUpload:

    int robot_basic_state; int robot_gait_state; int robot_policy_state;
    double rpy[3]; double rpy_vel[3]; double xyz_acc[3];
    double pos_world[3]; double vel_world[3]; double vel_body[3];
    unsigned touch_down_and_stair_trot; bool is_charging; unsigned error_state;
    int robot_motion_state; double battery_level; int task_state;
    bool is_robot_need_move; bool zero_position_flag; bool is_after_first_start; bool is_voice_ctrl_enable;
    double ultrasound[2];

A packet is the usual 12-byte head (uint32 code, uint32 parameters_size, uint32 type) followed by that struct. The
struct is parsed with natural C alignment (208 bytes) or packed (201 bytes), whichever the payload size says.

State tuple used everywhere: (basic, gait, policy, motion) = (robot_basic_state, robot_gait_state,
robot_policy_state, robot_motion_state). The doc's table writes it as (basic, gait-or-policy, motion): gait while
standing / walking (basic 6), policy in AI mode (basic 16).
"""
import struct
from typing import NamedTuple, Optional

STATE_CODE = 0x0901
_HEAD = struct.Struct('<3I')
# natural C alignment (x86-64 / aarch64): 4 bytes padding before rpy, 3 after is_charging
_ALIGNED = struct.Struct('<3i4x18dI?3xIidi4?2d')
_PACKED = struct.Struct('<3i18dI?Iidi4?2d')

BASIC_STANDING = 6       # torque control: standing / stepping in a normal gait
BASIC_AI = 16            # AI state
AI_POLICY_BASIC = 0      # robot_policy_state: AI basic gait

# robot_gait_state for each /simple_cmd gait code (doc sec 1.3.1)
GAIT_STATE_BY_CODE = {
    0x21010300: 0,       # Flat, Slow gear
    0x21010307: 4,       # Flat, Medium gear
    0x21010303: 5,       # Flat, Fast gear
    0x21010401: 2,       # RUG, General gear
    0x21010402: 6,       # RUG, Grip gear
    0x21010407: 13,      # RUG, H-Step gear
}
AI_ENTER_CODE = 0x21010528
AI_EXIT_CODE = 0x2101052B

STATE_NAMES = {
    (1, 0, 0): 'sitting', (1, 0, 11): 'long jump', (4, 0, 0): 'prepare', (5, 0, 0): 'sit-to-stand',
    (6, 0, 0): 'standing, flat slow', (6, 0, 1): 'stepping, flat slow', (6, 0, 2): 'twist', (6, 0, 4): 'twist jump',
    (6, 2, 0): 'standing, RUG general', (6, 2, 1): 'stepping, RUG general',
    (6, 4, 0): 'standing, flat medium', (6, 4, 1): 'stepping, flat medium',
    (6, 5, 0): 'standing, flat fast', (6, 5, 1): 'stepping, flat fast',
    (6, 6, 0): 'standing, RUG grip', (6, 6, 1): 'stepping, RUG grip', (6, 12, 1): 'moonwalk',
    (6, 13, 0): 'standing, RUG H-step', (6, 13, 1): 'stepping, RUG H-step',
    (7, 0, 0): 'sitting', (8, 0, 0): 'lose-control protection', (9, 0, 0): 'posture adjustment',
    (11, 0, 0): 'flipping over', (16, 0, 1): 'AI basic gait', (16, 16, 1): 'AI jump gait',
    (16, 18, 1): 'AI stand gait', (16, 20, 1): 'AI fast gait', (17, 0, 0): 'reset to zero',
    (18, 0, 0): 'backflip', (20, 0, 0): 'hello',
}


class RobotState(NamedTuple):
    basic: int
    gait: int
    policy: int
    motion: int
    battery: float = float('nan')
    error: int = 0

    @property
    def mode(self):
        return (self.basic, self.gait, self.policy, self.motion)


def describe(s) -> str:
    """'(6,13,1) stepping, RUG H-step' for a RobotState or a (basic, gait, policy, motion) tuple."""
    if s is None:
        return 'no robot state'
    basic, gait, policy, motion = tuple(s)[:4]
    mid = policy if basic == BASIC_AI else gait
    name = STATE_NAMES.get((basic, mid, motion), '')
    return f'({basic},{mid},{motion}){" " + name if name else ""}'


def parse(packet: bytes) -> Optional[RobotState]:
    """A 0x0901 robot state UDP packet -> RobotState; None for any other packet."""
    if len(packet) < _HEAD.size:
        return None
    code, size, _type = _HEAD.unpack_from(packet)
    if code != STATE_CODE:
        return None
    body = packet[_HEAD.size:]
    for layout in (_ALIGNED, _PACKED):
        if size == layout.size or (len(body) == layout.size):
            if len(body) < layout.size:
                return None
            v = layout.unpack_from(body)
            # 0-2 ints, 3-20 doubles, 21 touch_down, 22 is_charging, 23 error, 24 motion, 25 battery, ...
            return RobotState(basic=v[0], gait=v[1], policy=v[2], motion=v[24], battery=v[25], error=v[23])
    return None


def checkable(code: int) -> bool:
    """True if the robot state tells whether /simple_cmd `code` took effect."""
    return code in (AI_ENTER_CODE, AI_EXIT_CODE) or code in GAIT_STATE_BY_CODE


def target_reached(code: int, s, ai_policy: int = AI_POLICY_BASIC) -> Optional[bool]:
    """Is the robot (s = (basic, gait, policy, motion)) in the state that /simple_cmd `code` asks for?

    None = the code has no known target state (e.g. Move Mode, flat crawl): it cannot be checked.
    Gait codes: standing / stepping (basic 6) in that gait. Enter AI: basic 16 with the AI basic policy (16,0,1).
    Exit AI: back in torque control, basic 6 (any gait; the flat gait is sent and checked right after)."""
    if s is None:
        return None
    basic, gait, policy, _motion = tuple(s)[:4]
    if code == AI_ENTER_CODE:
        return basic == BASIC_AI and (ai_policy < 0 or policy == ai_policy)
    if code == AI_EXIT_CODE:
        return basic == BASIC_STANDING
    if code in GAIT_STATE_BY_CODE:
        return basic == BASIC_STANDING and gait == GAIT_STATE_BY_CODE[code]
    return None
