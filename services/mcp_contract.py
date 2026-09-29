"""Shared capability contract; lifecycle implementation remains mode-specific.

This module is pure configuration: no simulator, process-global mode, or secrets.
Discovery and dispatch use the same immutable session contract.
"""
from copy import deepcopy
from dataclasses import dataclass

ROBOT_TOOLS = frozenset({
    'robodojo_status', 'robodojo_observe', 'robodojo_pixel_to_position',
    'robodojo_pose_math', 'robodojo_step', 'robodojo_step_ee', 'robodojo_step_eef',
    'robodojo_free_space_move', 'robodojo_execute_motion_plan', 'robodojo_fk_preview',
})
PROFILES = ('rgbd', 'rgb-only', 'official')
# Official RoboDojo/XPolicyLab evaluation gives a policy only get_obs/take_action, whose
# actions are joint targets or native EEF poses (EvalEnv's own IK). Planners, pose math,
# previews and APIs must ship in the bundle.
OFFICIAL_ROBOT_TOOLS = frozenset({'robodojo_status', 'robodojo_observe', 'robodojo_step', 'robodojo_step_ee'})
# Session field carrying the last commanded grippers (the official state semantics).
COMMANDED_GRIPPERS = 'commanded_gripper_openings'


class Rejected(Exception):
    """A request refused before dispatch: nothing reached the robot."""


class ToolUnavailable(Rejected, PermissionError):
    pass


class InvalidArguments(Rejected, ValueError):
    pass


@dataclass(frozen=True)
class Mode:
    workspace: str
    lifecycle: frozenset
    episode_tools: frozenset
    gemini: bool = False
    multi_env: bool = False


MODES = {
    'auto-research': Mode('auto_research_agent',
        frozenset({'exploration_status', 'start_episode', 'evaluate', 'finish', 'register', 'rehearse', 'submit'}),
        # gemini_generate: official evaluation reaches it through a self-hosted remote
        # policy server (the AgentBundle bridge); the harness brokers it on the host.
        frozenset({'start_episode', 'evaluate', 'finish'}), gemini=True, multi_env=True),
}
GEOMETRY_KEYS = frozenset({
    'depth', 'camera_parameters', 'intrinsic_matrix', 'extrinsic_matrix',
    'camera_to_world_usd', 'cam_high_depth_m', 'cam_left_wrist_depth_m',
    'cam_right_wrist_depth_m',
})


@dataclass(frozen=True)
class ObservationProfile:
    name: str = 'rgbd'

    def __post_init__(self):
        if self.name not in PROFILES:
            raise ValueError('observation_profile must be rgbd, rgb-only or official')

    @property
    def geometry(self):
        return self.name == 'rgbd'

    @property
    def official(self):
        return self.name == 'official'

    def tools(self, definitions):
        tools = deepcopy(definitions)
        if self.geometry:
            return tools
        result = []
        for tool in tools:
            if tool['name'] == 'robodojo_pixel_to_position':
                continue
            if self.official and tool['name'] in ROBOT_TOOLS - OFFICIAL_ROBOT_TOOLS:
                continue
            tool['description'] = tool.get('description', '').replace('RGB/depth', 'RGB').replace(
                'attachments, calibration,', 'attachments (no depth or calibration),')
            result.append(tool)
        return result

    def clean(self, value):
        """Filter one observation tree; the commanded-gripper field never leaves this layer."""
        if isinstance(value, dict):
            if COMMANDED_GRIPPERS in value:
                value = dict(value)
                commanded = value.pop(COMMANDED_GRIPPERS)
                import math
                commanded = None if commanded is None else [float(x) for x in commanded]
                if (self.official and value.get('states') is not None and commanded is not None
                        and len(commanded) == 2 and all(math.isfinite(x) for x in commanded)):
                    # Official observations report the last commanded gripper, not the measured one.
                    states = [float(x) for x in value['states']]
                    states[6], states[13] = float(commanded[0]), float(commanded[1])
                    value['states'] = states
            return {key: self.clean(item) for key, item in value.items()
                    if self.geometry or str(key).lower() not in GEOMETRY_KEYS}
        if isinstance(value, (list, tuple)):
            return [self.clean(item) for item in value
                    if self.geometry or not (isinstance(item, dict) and item.get('kind') == 'depth')]
        return value


@dataclass(frozen=True)
class Contract:
    mode: str = 'auto-research'
    phase: str = 'exploration'
    observation_profile: str = 'official'
    environments: int = 1

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError('Unknown MCP mode')
        if self.phase not in {'exploration', 'isolated'}:
            raise ValueError('Unknown MCP phase')
        ObservationProfile(self.observation_profile)
        if type(self.environments) is not int or not 1 <= self.environments <= 8:
            raise ValueError('environments must be an integer in 1..8')
        if self.environments != 1 and (self.phase != 'exploration' or not MODES[self.mode].multi_env):
            raise ValueError('Multiple environments are exploration-only')

    @classmethod
    def from_config(cls, config, *, isolated=False):
        return cls(getattr(config, 'mode', 'auto-research'),
                   'isolated' if isolated else 'exploration',
                   getattr(config, 'observation_profile', 'official'),
                   1 if isolated else getattr(config, 'exploration_envs', 1))

    @property
    def observation(self):
        return ObservationProfile(self.observation_profile)

    @property
    def robot_names(self):
        if self.observation.official:
            return OFFICIAL_ROBOT_TOOLS
        return ROBOT_TOOLS if self.observation.geometry else ROBOT_TOOLS - {'robodojo_pixel_to_position'}

    @property
    def gemini(self):
        return MODES[self.mode].gemini

    @property
    def scoped_tools(self):
        return (self.robot_names - {'robodojo_pose_math'}) | MODES[self.mode].episode_tools

    def select(self, definitions, *, scoped=None):
        scoped = self.scoped_tools if scoped is None else scoped
        tools = self.observation.tools(definitions)
        selected = []
        # Isolated rehearsal/formal runs get robot tools only, no lifecycle tools.
        lifecycle = MODES[self.mode].lifecycle if self.phase == 'exploration' else frozenset()
        allowed = self.robot_names | lifecycle | ({'gemini_generate'} if self.gemini else set())
        for tool in tools:
            name = tool['name']
            if name not in allowed:
                continue
            if name in scoped and self.environments > 1:
                schema = tool['inputSchema']
                schema.setdefault('properties', {})['env_id'] = {
                    'type': 'integer', 'minimum': 0, 'maximum': self.environments - 1,
                    'description': 'Exploration environment ID; calls serialize within each environment.'}
                if 'env_id' not in schema.setdefault('required', []):
                    schema['required'].append('env_id')
                if name == 'robodojo_request_human_setup':
                    tool['description'] = 'Start/reset only the selected exploration environment; consumes one shared episode. The host chooses the seed.'
                suffix = ' Select env_id explicitly; all environments share the exploration episode budget.'
                if not tool['description'].endswith(suffix):
                    tool['description'] += suffix
            selected.append(tool)
        return selected

    def robot_definitions(self):
        from services.robodojo.mcp_server import RoboDojoMCP
        tools = [t for t in RoboDojoMCP.tool_definitions()
                 if t['name'] in ROBOT_TOOLS]
        return self.select(tools, scoped=self.robot_names - {'robodojo_pose_math'})

    def require_robot(self, name, arguments):
        if name not in self.robot_names and not (name == 'gemini_generate' and self.gemini):
            raise ToolUnavailable('Tool is not available in this MCP contract')
        if not isinstance(arguments, dict):
            raise InvalidArguments('Arguments must be an object')
        scoped = name in self.robot_names - {'robodojo_pose_math'}
        if scoped and self.environments > 1:
            env_id = arguments.get('env_id')
            if type(env_id) is not int or not 0 <= env_id < self.environments:
                raise InvalidArguments('A valid explicit env_id is required')
        elif 'env_id' in arguments:
            raise InvalidArguments('env_id is not available for this tool/context')

    def route(self, name, arguments):
        """Validate and remove the host routing field before a native worker call."""
        if name not in self.scoped_tools:
            raise ToolUnavailable('Tool is not available as an environment-scoped call')
        if not isinstance(arguments, dict):
            raise InvalidArguments('Arguments must be an object')
        payload = dict(arguments)
        if self.environments == 1:
            if 'env_id' in payload:
                raise InvalidArguments('env_id is not available in a single environment')
            return 0, payload
        env_id = payload.pop('env_id', None)
        if type(env_id) is not int or not 0 <= env_id < self.environments:
            raise InvalidArguments('A valid explicit env_id is required')
        return env_id, payload

    def manifest(self):
        return {'mode': self.mode, 'phase': self.phase, 'observation_profile': self.observation_profile,
                'exploration_envs': self.environments, 'robot_tools': sorted(self.robot_names),
                'gemini': self.gemini}

def observation_for(service):
    """Native services default to RGBD; launchers supply an immutable profile."""
    import os
    return ObservationProfile(getattr(service, 'observation_profile',
                                     os.environ.get('ROBODOJO_OBSERVATION_PROFILE', 'rgbd')))
