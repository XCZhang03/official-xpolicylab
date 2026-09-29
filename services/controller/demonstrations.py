"""Host-side visual context for offline research workspaces, never simulator data."""
from services.robodojo.demonstrations import (
    DEMONSTRATION_CONTEXTS, ensure_official_demo, ensure_terminal_frame,
    provision_trial_demonstration,
)
from .storage import persist


def cache_demonstration(cache_root, task, kind):
    if kind not in DEMONSTRATION_CONTEXTS:
        raise ValueError('Invalid demonstration_context')
    if kind == 'none':
        return None, None
    video = ensure_official_demo(cache_root, task)['path']
    terminal = ensure_terminal_frame(cache_root, task, video_path=video)['path'] if kind == 'terminal_state' else None
    return video, terminal


def provision_demonstration(config, workspace, cache_root):
    video, terminal = cache_demonstration(cache_root, config.task, config.demonstration_context)
    context = provision_trial_demonstration(video_path=video, terminal_path=terminal,
        target=workspace/'runtime/demonstrations', task=config.task, context_kind=config.demonstration_context)
    # Keep the trusted descriptor separate from agent-writable image/manifest copies.
    persist(config.root/'demonstration-context.json', context)
    return context


def task_demonstration_context(context):
    if context['kind'] == 'none':
        return '\n## Visual demonstration\n\nNo visual demonstration was selected for this session.\n'
    result = (f"\n## Visual demonstration\n\nSelected: `{context['kind']}`; {context['image_count']} image(s).\n"
            "Read [the image manifest](runtime/demonstrations/manifest.json) and view the\n"
            "local images before acting. `ordered_images` paths are relative to\n"
            "`/workspace/runtime/demonstrations/`; preserve their order for a sequence.\n"
            "Use the in-context-action-learning skill. These images explain task goals;\n"
            "their layout may differ from the current scene. They contain no executable\n"
            "actions or simulator state. Fulfill the rubric in the current layout.\n"
            "The images are development context, not automatically mounted in isolated\n"
            "runs. Include any reference images needed by your script in its bundle.\n")
    return result
