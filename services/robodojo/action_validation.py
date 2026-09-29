"""Validate complete motion requests before any simulator RPC can advance physics."""
import numpy as np

from .kinematics import validate_eef_target
from .trajectory import MAX_TRAJECTORY_ACTIONS


class ActionValidationError(ValueError):
    """An agent-correctable rejection with a guarantee that no action was sent."""


def validate_ee_rows(rows):
    """16-D official EEF rows: [left x,y,z,qw,qx,qy,qz,gripper, right ...]."""
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 16 or not np.isfinite(matrix).all():
        raise ValueError('Each EEF action must be a finite 16-D vector: '
                         '[left x,y,z,qw,qx,qy,qz,gripper, right x,y,z,qw,qx,qy,qz,gripper]')
    if np.any((matrix[:, (7, 15)] < 0) | (matrix[:, (7, 15)] > 1)):
        raise ValueError('Gripper openings must be in [0, 1]')
    norms = np.linalg.norm(matrix[:, [3, 4, 5, 6, 11, 12, 13, 14]].reshape(-1, 2, 4), axis=2)
    if np.any(np.abs(norms - 1) > 1e-3):
        raise ValueError('EEF quaternions must be unit wxyz (normalize them)')
    return matrix


def validate_motion_request(name, arguments):
    if name not in {'robodojo_step', 'robodojo_step_ee', 'robodojo_step_eef',
                    'robodojo_free_space_move', 'robodojo_execute_motion_plan'}:
        return
    if not isinstance(arguments, dict):
        raise ActionValidationError('Motion arguments must be an object; no action executed.')
    try:
        if name in {'robodojo_step', 'robodojo_step_ee', 'robodojo_step_eef'}:
            field = 'targets' if name == 'robodojo_step_eef' else 'actions'
            if set(arguments) != {field}:
                raise ValueError(f'{name} requires exactly the {field} field')
            rows = arguments[field]
            if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TRAJECTORY_ACTIONS:
                count = len(rows) if isinstance(rows, list) else 'a non-list value'
                raise ValueError(f'{field} must contain 1..{MAX_TRAJECTORY_ACTIONS} rows; received {count}. Split longer sequences into calls')
            if name == 'robodojo_step':
                matrix = np.asarray(rows, dtype=np.float32)
                if matrix.shape != (len(rows), 14) or not np.isfinite(matrix).all():
                    raise ValueError('Each action must be a finite 14-D vector')
                if np.any((matrix[:, (6, 13)] < 0) | (matrix[:, (6, 13)] > 1)):
                    raise ValueError('Gripper openings must be in [0, 1]')
            elif name == 'robodojo_step_ee':
                validate_ee_rows(rows)
            else:
                for index, target in enumerate(rows):
                    if not isinstance(target, dict) or set(target) != {'left', 'right'}:
                        raise ValueError(f'targets[{index}] must explicitly contain left and right')
                    for arm in ('left', 'right'):
                        try:
                            validate_eef_target(target[arm], require_gripper=True)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'targets[{index}].{arm}: {exc}') from exc
        elif name == 'robodojo_free_space_move':
            if set(arguments) - {'arm', 'target', 'preview_only', 'include_trajectory'}:
                raise ValueError('Unknown free_space_move fields; use arm, target, preview_only and include_trajectory only')
            if arguments.get('arm') not in ('left', 'right'):
                raise ValueError('arm must be left or right')
            if any(type(arguments.get(key, False)) is not bool for key in ('preview_only', 'include_trajectory')):
                raise ValueError('preview_only and include_trajectory must be boolean')
            validate_eef_target(arguments.get('target'))
        elif set(arguments) != {'motion_plan_id'} or not isinstance(arguments.get('motion_plan_id'), str) or not arguments['motion_plan_id']:
            raise ValueError('Provide exactly one nonempty string motion_plan_id')
    except (ValueError, TypeError, OverflowError) as exc:
        raise ActionValidationError(f'{exc}; no action executed.') from exc
