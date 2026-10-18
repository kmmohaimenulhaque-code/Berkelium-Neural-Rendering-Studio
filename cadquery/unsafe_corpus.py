"""Injected unsafe CadQuery scripts. D-034 requires the sandbox to reject 100 % of them.

Each one is a plausible model output with a payload: file access, process spawn, network, interpreter escape via
dunders, dynamic code, CadQuery's own file exporters/importers, or resource abuse. "Rejected" means the sandbox
returned ``ok=False`` and the payload had no effect (``tests/test_sandbox.py`` also checks the canary file)."""

_BOX = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10)\n"
CANARY = "/tmp/berkelium_sandbox_canary"

UNSAFE: dict[str, str] = {
    "import_os": "import os\n" + _BOX + f"os.system('touch {CANARY}')\n",
    "import_sys": "import sys\n" + _BOX + "sys.exit(0)\n",
    "import_subprocess": "import subprocess\n" + _BOX + f"subprocess.run(['touch', '{CANARY}'])\n",
    "from_os": "from os import system\n" + _BOX + f"system('touch {CANARY}')\n",
    "import_socket": "import socket\n" + _BOX + "socket.create_connection(('1.1.1.1', 80))\n",
    "import_urllib": "import urllib.request\n" + _BOX + "urllib.request.urlopen('http://example.com')\n",
    "import_shutil": "import shutil\n" + _BOX + "shutil.rmtree('/tmp')\n",
    "import_ctypes": "import ctypes\n" + _BOX,
    "import_math_submodule_trick": "import math.os\n" + _BOX,
    "open_write": _BOX + f"open('{CANARY}', 'w').write('x')\n",
    "open_read_secrets": _BOX + "print(open('/etc/passwd').read())\n",
    "exec": _BOX + f"exec(\"import os; os.system('touch {CANARY}')\")\n",
    "eval": _BOX + "eval(\"1+1\")\n",
    "compile": _BOX + "compile('1', 'x', 'eval')\n",
    "dunder_import": _BOX + f"__import__('os').system('touch {CANARY}')\n",
    "builtins_dunder": _BOX + "__builtins__['open']('/etc/passwd')\n",
    "class_mro_escape": _BOX + "().__class__.__mro__[1].__subclasses__()\n",
    "getattr_escape": _BOX + "getattr((), '__class__')\n",
    "globals": _BOX + "globals()['__builtins__']\n",
    "vars": _BOX + "vars()\n",
    "func_globals": "import cadquery as cq\ndef f():\n    pass\nf.__globals__\nresult = cq.Workplane().box(1, 1, 1)\n",
    "string_dunder": _BOX + "x = '__class__'\n",
    "lambda_dunder": _BOX + "(lambda: 0).__code__\n",
    "cq_exporters": _BOX + f"cq.exporters.export(result, '{CANARY}.step')\n",
    "cq_export_method": _BOX + f"result.export('{CANARY}.step')\n",
    "cq_importers": "import cadquery as cq\nresult = cq.importers.importStep('/etc/passwd')\n",
    "cq_save": "import cadquery as cq\na = cq.Assembly()\na.save('/tmp/x.step')\nresult = cq.Workplane().box(1, 1, 1)\n",
    "cq_occ_impl": "import cadquery as cq\nresult = cq.occ_impl.shapes.Solid.makeBox(1, 1, 1)\n",
    "cq_wrapped": _BOX + "w = result.val().wrapped\n",
    "from_cq_import_exporters": "from cadquery import exporters\n" + _BOX,
    "star_import": "from cadquery import *\nresult = Workplane().box(1, 1, 1)\n",
    "ocp_import": "import OCP\n" + _BOX,
    "breakpoint": _BOX + "breakpoint()\n",
    "input": _BOX + "input()\n",
    "type_metaclass": _BOX + "type('X', (), {})\n",
    "class_def": "import cadquery as cq\nclass X:\n    pass\nresult = cq.Workplane().box(1, 1, 1)\n",
    "global_stmt": "import cadquery as cq\ndef f():\n    global result\nresult = cq.Workplane().box(1, 1, 1)\n",
    "no_result": "import cadquery as cq\nr = cq.Workplane().box(1, 1, 1)\n",
    "relative_import": "from . import x\n" + _BOX,
    "infinite_loop": "import cadquery as cq\nwhile True:\n    pass\nresult = cq.Workplane().box(1, 1, 1)\n",
    "memory_bomb": "import cadquery as cq\nx = [0] * (10 ** 10)\nresult = cq.Workplane().box(1, 1, 1)\n",
    "cpu_bomb": "import cadquery as cq\nx = 9 ** 9 ** 9\nresult = cq.Workplane().box(1, 1, 1)\n",
}
