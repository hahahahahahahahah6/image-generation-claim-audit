"""Offline documentation and Python syntax inspection."""
import ast
import fnmatch
import os
import re
from pathlib import Path

from .phrases import IMAGEGENERATION, NOIMAGEGENERATION

PIPELINES = frozenset('StableDiffusionPipeline StableDiffusionXLPipeline StableDiffusion3Pipeline StableDiffusionImg2ImgPipeline StableDiffusionInpaintPipeline StableDiffusionControlNetPipeline DiffusionPipeline AutoPipelineForText2Image AutoPipelineForImage2Image FluxPipeline KandinskyPipeline DDPMPipeline DDIMPipeline'.split())


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            lines.append('')
        elif marker:
            fence = marker[1]
            lines.append('')
        else:
            lines.append(line)
    return '\n'.join(lines)


def matches(text, phrases):
    spans = []
    for phrase in sorted(phrases, key=lambda s: (-len(s), s)):
        for match in re.finditer(r'\b' + re.escape(phrase) + r'\b', text):
            if not any(match.start() < b and match.end() > a for a, b in spans):
                spans.append(match.span())
    return spans


def claims(text):
    positive = negative = False
    for paragraph in re.split(r'\n\s*\n', prose(text)):
        paragraph = normalize(paragraph)
        spans = matches(paragraph, NOIMAGEGENERATION)
        negative |= bool(spans)
        chars = list(paragraph)
        for a, b in spans:
            chars[a:b] = ' ' * (b-a)
        positive |= bool(matches(''.join(chars), IMAGEGENERATION))
    return positive, negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False

    def bindings(body, inherited):
        env = dict(inherited)
        # Bindings belong to one lexical scope; nested bodies are visited separately.
        def bind(node):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    env[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    env[alias.asname or alias.name] = (node.module or '') + '.' + alias.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                env[node.name] = None
                return
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                        env[child.id] = None
                    bind(child)
        for node in body:
            bind(node)
        # A local definition always shadows an imported spelling in this scope.
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                env[node.name] = None
        return env

    def resolve(node, env):
        if isinstance(node, ast.Name):
            return env.get(node.id) or ''
        if isinstance(node, ast.Attribute):
            base = resolve(node.value, env)
            return base + '.' + node.attr if base else ''
        return ''

    def pipeline(node, env):
        name = resolve(node, env)
        return name.startswith('diffusers.') and name.rsplit('.', 1)[-1] in PIPELINES

    def walk(node, env):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner = bindings(node.body, env)
            if hasattr(node, 'args'):
                args = node.args
                for arg in args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]:
                    inner[arg.arg] = None
            return any(walk(child, inner) for child in node.body)
        if isinstance(node, ast.Call):
            func = node.func
            if pipeline(func, env):
                return True
            if isinstance(func, ast.Attribute):
                if func.attr in ('from_pretrained', 'from_single_file') and pipeline(func.value, env):
                    return True
                if func.attr == 'generate' and isinstance(func.value, ast.Attribute) and func.value.attr == 'images':
                    return True
            if resolve(func, env) == 'transformers.pipeline':
                tasks = list(node.args[:1]) + [k.value for k in node.keywords if k.arg == 'task']
                if any(isinstance(t, ast.Constant) and isinstance(t.value, str) and any(p in normalize(t.value) for p in ('text to image', 'image generation')) for t in tasks):
                    return True
        return any(walk(child, env) for child in ast.iter_child_nodes(node))

    return walk(tree, bindings(tree.body, {}))


def audit(root, ignore=()):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('root must be a readable directory')
    positive = negative = production = test = False
    count = 0
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if not (Path(directory)/d).is_symlink())
        for name in sorted(files):
            path = Path(directory)/name
            relative = path.relative_to(root)
            rel = relative.as_posix()
            if path.is_symlink() or any(fnmatch.fnmatchcase(rel, pattern) for pattern in ignore):
                continue
            parts = relative.parts
            doc = rel == 'README.md' or (path.suffix == '.md' and 'docs' in parts[:-1])
            source = path.suffix == '.py' and not {'fixtures', 'docs', 'examples'}.intersection(parts[:-1])
            if not (doc or source):
                continue
            text = path.read_text(encoding='utf-8')
            count += 1
            if doc:
                p, n = claims(text)
                positive |= p
                negative |= n
            if source and evidence(text):
                if 'tests' in parts[:-1] or name.startswith('test_') or name.endswith('_test.py'):
                    test = True
                else:
                    production = True
    findings = []
    if positive and not production:
        findings.append(dict(rule='documented-image-generation-claim-false', severity='MEDIUM' if test else 'HIGH', message='Documentation promises picture synthesis without production evidence.'))
    if negative and production:
        findings.append(dict(rule='documented-no-image-generation-claim-false', severity='HIGH', message='Documentation denies picture synthesis but production evidence exists.'))
    return dict(verdict='FAIL' if any(f['severity'] == 'HIGH' for f in findings) else 'PASS', findings=findings, files_scanned=count)
