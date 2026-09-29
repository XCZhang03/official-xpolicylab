"""Collect exactly the RoboDojo definitions one task depends on, as source excerpts.

The task module (and any task module it imports) is copied whole. Everything else is
excerpted per definition, starting from what the task code actually uses:

- **reward checks**: each ``RewardManager`` method the task calls, the
  ``Func_Parser`` implementation it dispatches to (by name), and every helper those
  reach (``self.<method>``, module functions, constants, and names imported from
  other RoboDojo modules such as ``utils.transformer``), transitively;
- **episode flow**: the official ``EvalEnv`` methods that step actions, enforce the
  step limit, run the success and score checks and end the episode;
- **framework calls**: methods the task calls on itself or on ``robot_manager`` that
  it does not define (e.g. support-arm trajectory helpers), excerpted one level deep;
- names the task module imports with ``from <robodojo module> import ...``.

Scene, camera, robot and simulator machinery is not included beyond those methods.
"""
from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROBODOJO = Path(__file__).resolve().parents[1] / "RoboDojo"
TASKS = ROBODOJO / "task/RoboDojo/tasks"
REWARD = ROBODOJO / "env/reward_manager/reward_manager.py"
PARSER = ROBODOJO / "env/reward_manager/func_parser.py"
EVAL_ENV = ROBODOJO / "src/eval_client/eval_env.py"
FRAMEWORK = {  # self.<attr>.<method>(...) receivers and the classes they hold
    "robot_manager": (ROBODOJO / "env/robot_manager/robot_manager.py", "RobotManager"),
    "reward_manager": (REWARD, "RewardManager"),
}
SELF_CLASSES = [(EVAL_ENV, None), (ROBODOJO / "env/environment/task_env.py", "TaskEnv"),
                (ROBODOJO / "env/environment/base_env.py", "BaseEnv")]
EPISODE_FLOW = ("reset", "take_action", "_step_limit_reached", "_step_progress", "take_action_batch",
                "run_eval", "is_episode_end", "mark_env_unstable")
REWARD_CORE = ("__init__", "reset", "initialize", "init_state", "step", "call_func_parser",
               "get_reward", "get_score")
PARSER_CORE = ("__init__", "reset", "initialize", "init_state", "_check_env_success")


def module_name(path):
    return ".".join(Path(path).relative_to(ROBODOJO).with_suffix("").parts)


def module_file(name):
    path = ROBODOJO.joinpath(*name.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


class Module:
    def __init__(self, path):
        self.path = Path(path)
        self.source = self.path.read_text()
        self.tree = ast.parse(self.source)
        self.top, self.classes, self.imports, self.star = {}, {}, {}, []
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.top[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            self.top[name.id] = node
            elif isinstance(node, ast.ImportFrom) and node.module and module_file(node.module):
                for alias in node.names:
                    if alias.name == "*":
                        self.star.append(node.module)
                    else:
                        self.imports[alias.asname or alias.name] = (node.module, alias.name)
        # Classes anywhere (EvalEnv is defined inside create_eval_env).
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                self.classes.setdefault(node.name, {n.name: n for n in node.body
                                                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
                self.classes[node.name]["__class__"] = node

    def segment(self, node):
        """Source of ``node`` including decorators, dedented."""
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        return textwrap.dedent("\n".join(self.source.splitlines()[start - 1:node.end_lineno]))

    def header_imports(self):
        return [ast.get_source_segment(self.source, n) for n in self.tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom))]


_MODULES = {}


def load(path):
    path = Path(path)
    if path not in _MODULES:
        _MODULES[path] = Module(path)
    return _MODULES[path]


def references(node):
    names, attributes, strings = set(), set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == "self":
            attributes.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            strings.add(child.value)
    return names, attributes, strings


class Collector:
    """Worklist of (module path, class or None, name) definitions to excerpt."""

    def __init__(self):
        self.wanted, self.todo = {}, []

    def want(self, path, cls, name, *, deep=True):
        if cls is None and isinstance(load(path).top.get(name), ast.ClassDef):
            return  # A class is excerpted only through the methods that are used.
        key = (Path(path), cls, name)
        if key in self.wanted:
            self.wanted[key] = self.wanted[key] or deep
            return
        self.wanted[key] = deep
        self.todo.append(key)

    def resolve_name(self, module, name):
        """(path, None, name) for a module-level name visible in ``module``."""
        if name in module.top:
            return module.path, None, name
        if name in module.imports:
            target, original = module.imports[name]
            return module_file(target), None, original
        for star in module.star:
            other = load(module_file(star))
            if name in other.top:
                return other.path, None, name
        return None

    def run(self):
        parser = load(PARSER)
        parser_methods = set(parser.classes["Func_Parser"])
        while self.todo:
            path, cls, name = self.todo.pop()
            deep = self.wanted[(path, cls, name)]
            module = load(path)
            if cls is None:
                node = module.top.get(name)
            else:
                node = module.classes.get(cls, {}).get(name)
            if node is None or not deep:
                continue
            names, attributes, strings = references(node)
            for ref in names - {name}:
                found = self.resolve_name(module, ref)
                if found and found[0] is not None:
                    self.want(*found)
            if cls is not None:
                for attribute in attributes & set(module.classes[cls]):
                    self.want(path, cls, attribute)
            if path == REWARD:
                for string in strings & parser_methods:  # ("is_A_in_B", args) dispatch by name
                    self.want(PARSER, "Func_Parser", string)


def task_modules(task):
    """The task module plus task modules it imports (e.g. a _random variant's base)."""
    seen, todo = [], [TASKS / f"{task}.py"]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.append(path)
        for alias_module in load(path).imports.values():
            file = module_file(alias_module[0])
            if file and file.parent == TASKS and file not in seen:
                todo.append(file)
        for star in load(path).star:
            file = module_file(star)
            if file and file.parent == TASKS and file not in seen:
                todo.append(file)
    return seen


def collect(task):
    """{relative path: excerpt text} for the task's dependencies (task modules whole)."""
    collector = Collector()
    tasks = task_modules(task)
    reward = load(REWARD)
    for path in tasks:
        module = load(path)
        defined = {n for c in module.classes.values() for n in c}
        names, _, _ = references(module.tree)
        for ref in names:
            found = collector.resolve_name(module, ref)
            if found and found[0] is not None and found[0].parent != TASKS:
                collector.want(*found)
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                    and value.value.id == "self" and value.attr in FRAMEWORK):
                framework, cls = FRAMEWORK[value.attr]
                collector.want(framework, cls, node.attr, deep=framework == REWARD)
            elif isinstance(value, ast.Name) and value.id in ("rm", "reward_manager") \
                    and node.attr in reward.classes["RewardManager"]:
                collector.want(REWARD, "RewardManager", node.attr)
            elif (isinstance(value, ast.Name) and value.id == "self" and node.attr not in defined):
                for framework, cls in SELF_CLASSES:
                    classes = load(framework).classes
                    owner = cls or next((c for c, methods in classes.items() if node.attr in methods), None)
                    if owner and node.attr in classes.get(owner, {}):
                        collector.want(framework, owner, node.attr, deep=False)
                        break
    for name in REWARD_CORE:
        collector.want(REWARD, "RewardManager", name)
    for name in PARSER_CORE:
        collector.want(PARSER, "Func_Parser", name)
    eval_class = next(c for c, methods in load(EVAL_ENV).classes.items() if "run_eval" in methods)
    for name in EPISODE_FLOW:
        collector.want(EVAL_ENV, eval_class, name, deep=False)
    collector.run()
    files = {str(p.relative_to(ROBODOJO)): p.read_text() for p in tasks}
    by_path = {}
    for (path, cls, name) in collector.wanted:
        if path.parent == TASKS:
            continue
        by_path.setdefault(path, set()).add((cls, name))
    for path, entries in sorted(by_path.items()):
        files[str(path.relative_to(ROBODOJO))] = render(load(path), entries)
    return files


def render(module, entries):
    """An excerpt of ``module`` with the wanted definitions in source order."""
    order = []
    for cls, name in entries:
        node = module.top.get(name) if cls is None else module.classes.get(cls, {}).get(name)
        if node is not None:
            order.append((node.lineno, cls, name, node))
    order.sort(key=lambda row: row[0])
    lines = [f'"""Excerpt of RoboDojo {module.path.relative_to(ROBODOJO)}: only the definitions this task',
             'uses. Reference only (not runnable); line numbers refer to the original file."""', ""]
    lines += module.header_imports() + [""]
    current = None
    for lineno, cls, name, node in order:
        if cls != current:
            if cls is not None:
                class_node = module.classes[cls]["__class__"]
                bases = ", ".join(ast.get_source_segment(module.source, b) for b in class_node.bases)
                lines += ["", f"class {cls}{f'({bases})' if bases else ''}:  # excerpt"]
            current = cls
        body = module.segment(node).rstrip()
        comment = f"# {module.path.name}:{lineno}"
        if cls is not None:
            lines += ["", textwrap.indent(comment + "\n" + body, "    ")]
        else:
            lines += ["", comment, body]
    return "\n".join(lines) + "\n"
