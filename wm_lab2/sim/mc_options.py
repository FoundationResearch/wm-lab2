from __future__ import annotations

import os
import tempfile


def ensure_minedojo_autojump(*, enabled: bool = True) -> None:
    """
    Best-effort patch for MineDojo's bundled Minecraft options template.

    MineDojo ships a template file at:
      minedojo/sim/Malmo/Minecraft/run/options.txt
    which contains a line like:
      autoJump:false

    We flip it to `autoJump:true` so newly launched Minecraft instances inherit
    auto-jump, without requiring manual UI interaction.

    If MineDojo isn't installed, the file isn't present, or the environment is
    read-only, this function simply returns without raising.
    """
    if not bool(enabled):
        return

    try:
        import minedojo  # type: ignore

        pkg_dir = os.path.dirname(minedojo.__file__)
        options_path = os.path.join(pkg_dir, "sim", "Malmo", "Minecraft", "run", "options.txt")
        if not os.path.isfile(options_path):
            return

        with open(options_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        out_lines: list[str] = []
        found = False
        for line in lines:
            if line.startswith("autoJump:"):
                found = True
                out_lines.append("autoJump:true\n")
            else:
                out_lines.append(line)
        if not found:
            # Keep file readable; ensure it ends with newline.
            if out_lines and not out_lines[-1].endswith("\n"):
                out_lines[-1] = out_lines[-1] + "\n"
            out_lines.append("autoJump:true\n")

        if out_lines == lines:
            return

        # Atomic-ish replace: write temp file in same dir then os.replace.
        d = os.path.dirname(options_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=d, delete=False) as tf:
            tmp_path = tf.name
            tf.writelines(out_lines)
        os.replace(tmp_path, options_path)
    except Exception:
        # Best effort only.
        try:
            if "tmp_path" in locals() and isinstance(tmp_path, str) and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return


