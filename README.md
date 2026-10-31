# Prompt conditioned picture creation audit

An offline static checker for documented diffusion model picture synthesis.
It reads root README.md and Markdown beneath any docs directory. Paragraphs
are joined across line breaks; fenced examples are excluded and inline code
counts. Matching uses exact normalized templates, contractions, word boundaries,
and longest matches first. Negative spans are removed before positive matching.

Install and run with Python 3.9 or newer; no runtime dependencies are required.

```sh
pip install image-generation-claim-audit
image-generation-claim-audit ./project --json
python3 -m image_generation_claim_audit ./project --verbose
image-generation-claim-audit ./project --ignore 'tests/*' --fail-on MEDIUM
image-generation-claim-audit --version
```

The two rules report contradictory promises and denials respectively:

```text
documented-image-generation-claim-false
documented-no-image-generation-claim-false
```

A promise without production evidence is HIGH; evidence confined to tests lowers
it to MEDIUM. A denial contradicted by production evidence is HIGH. Each rule
appears at most once. Without a matching claim the result is PASS.
The default threshold is HIGH; selecting MEDIUM also fails on test-only evidence.
Exit codes are 0 for PASS, 2 for FAIL, and 1 for an invalid or unreadable root
or command usage error. JSON contains verdict, findings (rule, severity,
message), and files_scanned. Verbose human output includes the scanned count.
Repeatable ignore patterns use fnmatch against relative POSIX paths.

Evidence comes from alias-resolved diffusers pipeline construction or loading,
transformers task calls with a qualifying literal, and hosted picture API calls.
Pipeline loads through from_pretrained and from_single_file count without
execution. Hosted calls use the images.generate attribute shape with any
receiver. Local definitions shadow imported names. Bare imports, strings,
drawing, resizing, recognition, audio, and movie operations do not count.

The checker parses Python with ast and never runs inspected code. Symlinks are
excluded. Source beneath fixtures, docs, and examples is excluded; tests and
conventional test filenames are considered separately. Invalid syntax supplies
no evidence. files_scanned counts eligible documents and source files read,
including syntax-invalid source. Files that cannot be read produce exit 1.

This is a lexical, single-module approximation: it does not follow cross-module
aliases, prove execution paths, validate credentials, download weights, or check
picture quality. A qualifying construction alone suffices, including unreachable
code. Providers outside the recognized forms may need manual review.

Development uses the standard library unittest runner and isolated setuptools
builds. The source archive includes the suite and its fixtures. MIT licensed.

```sh
python3 -m unittest discover
python3 -m build
```
