"""Dark and light chrome colors. Data colors (tracks, fit lines, warnings) stay put."""

from __future__ import annotations

from pathlib import Path

_NAME = "dark"
_QSS = Path(__file__).with_name("style.qss")

# token -> (dark, light). Dark values are the colors already in the app.
PALETTE: dict[str, tuple[str, str]] = {
    "bg": ("#1f1f1f", "#f4f5f7"),
    "bg_status": ("#1a1a1a", "#eef0f2"),
    "bg_stage": ("#1a1c20", "#e8eaed"),
    "bg_pressed": ("#191919", "#d8dadd"),
    "bg_pressed_2": ("#222222", "#d8dadd"),
    "sunken": ("#1c1c1c", "#f7f8fa"),
    "panel": ("#232323", "#ffffff"),
    "panel_alt": ("#252525", "#f7f8fa"),
    "raised": ("#2a2a2a", "#f0f1f3"),
    "input": ("#2d2d2d", "#ffffff"),
    "input_read": ("#262626", "#f3f4f6"),
    "menu": ("#292929", "#ffffff"),
    "button": ("#303030", "#eef0f2"),
    "button_border": ("#303030", "#d5d5d5"),
    "button_hover": ("#3b3b3b", "#e4e6e9"),
    "button_pressed": ("#242424", "#dcdfe3"),
    "button_disabled": ("#2c2c2c", "#f3f4f6"),
    "button_disabled_border": ("#383838", "#d5d5d5"),
    "hover": ("#353535", "#e6e8eb"),
    "hover_input": ("#363636", "#f3f4f6"),
    "hover_tool": ("#383838", "#e4e6e9"),
    "hover_strong": ("#454545", "#dcdfe3"),
    "hover_soft": ("#373737", "#e6e8eb"),
    "selected": ("#3a3a3a", "#e6e8eb"),
    "toast": ("#2b2b2b", "#ffffff"),
    "toast_border": ("#3e3e3e", "#d5d5d5"),
    "chip": ("#2e2e2e", "#eef0f2"),
    "text": ("#dedede", "#1c1c1c"),
    "text_menu": ("#dddddd", "#1c1c1c"),
    "text_body": ("#d6d6d6", "#1c1c1c"),
    "text_soft": ("#d0d0d0", "#2a2a2a"),
    "text_bright": ("#eeeeee", "#1c1c1c"),
    "text_strong": ("#f0f0f0", "#1c1c1c"),
    "text_readout": ("#e8e8e8", "#1c1c1c"),
    "text_toast": ("#e6e6e6", "#1c1c1c"),
    "text_cal": ("#f5f5f5", "#1c1c1c"),
    "text_dim": ("#bdbdbd", "#4a4a4a"),
    "text_secondary": ("#b8b8b8", "#5c5c5c"),
    "text_muted": ("#8f8f8f", "#5c5c5c"),
    "text_faint": ("#8a8a8a", "#6a6a6a"),
    "text_check": ("#c8c8c8", "#2a2a2a"),
    "text_disabled": ("#707070", "#9a9a9a"),
    "text_disabled_2": ("#6a6a6a", "#9a9a9a"),
    "text_toast_dim": ("#a8a8a8", "#5c5c5c"),
    "border": ("#3a3a3a", "#d5d5d5"),
    "border_bar": ("#3c3c3c", "#d5d5d5"),
    "line": ("#333333", "#e1e3e6"),
    "border_input": ("#454545", "#c8ccd1"),
    "border_menu": ("#484848", "#d0d0d0"),
    "border_popup": ("#494949", "#d0d0d0"),
    "border_line": ("#4a4a4a", "#d0d3d8"),
    "border_btn": ("#444444", "#c8ccd1"),
    "border_play": ("#505050", "#c5c8cc"),
    "border_hover": ("#595959", "#b0b4b8"),
    "border_disabled": ("#393939", "#d5d5d5"),
    "border_loop": ("#5a5a5a", "#c5c8cc"),
    "border_indicator": ("#6a6a6a", "#b0b0b0"),
    "groove": ("#5a5a5a", "#c5c8cc"),
    "groove_fill": ("#9a9a9a", "#3a3a3a"),
    "selection": ("#5a5a5a", "#d4e4f4"),
    "selection_menu": ("#505050", "#d4e4f4"),
    "selection_list": ("#555555", "#d4e4f4"),
    "checked_bg": ("#3d4a3a", "#e5f0e4"),
    "checked_border": ("#5a6a52", "#7d9a72"),
    "loop_on": ("#e0e0e0", "#1c1c1c"),
    "accent": ("#6cb6ff", "#2f7dcc"),
    "accent_bright": ("#4da3ff", "#2b7fd0"),
    "teal": ("#80cbc4", "#1f8a82"),
    "loop_hover_border": ("#7a7a7a", "#8a8a8a"),
    "assistant_bg": ("#191815", "#f7f6f3"),
    "assistant_sunken": ("#161512", "#eef0f2"),
    "assistant_card": ("#1f1d19", "#ffffff"),
    "assistant_input": ("#23211c", "#ffffff"),
    "assistant_button": ("#2a2722", "#eef0f2"),
    "assistant_hover": ("#35322c", "#e6e8eb"),
    "assistant_primary_hover": ("#3a3731", "#e4e6e9"),
    "assistant_line": ("#2e2c28", "#e4e2de"),
    "assistant_border": ("#322f2a", "#d8dadd"),
    "assistant_border_2": ("#3a3834", "#d0d0d0"),
    "assistant_selection": ("#3f3c36", "#d6e4f2"),
    "assistant_text": ("#f0ece4", "#1c1c1c"),
    "assistant_muted": ("#b7b1a6", "#5c5c5c"),
    "assistant_faint": ("#9a958c", "#6a6a6a"),
    "assistant_think": ("#8a857c", "#6a6a6a"),
    "assistant_disabled": ("#7a756c", "#9a9a9a"),
    "assistant_link": ("#d4c4a8", "#5c5348"),
    "assistant_link_2": ("#c8c2b6", "#3a3a3a"),
    "assistant_focus": ("#8a8174", "#5c5c5c"),
    "chip_ok": ("#b7c4a4", "#2f6b3a"),
    "chip_ok_border": ("#3d4338", "#b7cfc0"),
    "chip_warn": ("#d4b27a", "#8a5a12"),
    "chip_warn_border": ("#4a3f2c", "#e2c48a"),
    "chip_bad": ("#c98978", "#a33b2b"),
    "chip_bad_border": ("#4a3530", "#e2b2a8"),
    "chart_label": ("#9a9a9a", "#5c5c5c"),
    "chart_title": ("#bdbdbd", "#3a3a3a"),
    "chart_grid": ("#2f2f2f", "#e4e6e9"),
    "chart_axis": ("#4a4a4a", "#b0b4b8"),
    "video_letterbox": ("#1f1f1f", "#e8eaed"),
    "drop_bg": ("#1b1b1b", "#f7f8fa"),
    "drop_bg_hover": ("#202020", "#eef1f4"),
    "drop_border": ("#474747", "#c5c8cc"),
    "drop_border_hover": ("#777777", "#8a8a8a"),
    "drop_icon": ("#737373", "#8a8a8a"),
    "drop_icon_hover": ("#a5a5a5", "#5c5c5c"),
    "drop_sub": ("#858585", "#6a6a6a"),
    "drop_title": ("#dddddd", "#1c1c1c"),
    "drop_status_bg": ("#292929", "#e4e6e9"),
    "drop_status": ("#c7c7c7", "#1c1c1c"),
    "icon_ink": ("#dddddd", "#2a2a2a"),
    "icon_muted": ("#c8c8c8", "#3a3a3a"),
    "icon_loop": ("#cccccc", "#2a2a2a"),
    "icon_on": ("#1d1d1d", "#f4f5f7"),
    "marker": ("#c4c4c4", "#3a3a3a"),
    "marker_off": ("#6a6a6a", "#b0b0b0"),
    "playhead": ("#e2e2e2", "#1c1c1c"),
    "playhead_off": ("#767676", "#b0b0b0"),
    "stage": ("#1a1c20", "#e8eaed"),
    "stage_floor": ("#24262c", "#dfe3e8"),
    "stage_deep": ("#16181c", "#d5d9de"),
    "stage_line": ("#3a3d44", "#c5c8cc"),
    "stage_ink": ("#e6e6e6", "#1c1c1c"),
    "stage_chip": ("#2a3340", "#d6e4f2"),
    "stage_accent_text": ("#b8d4ff", "#1d5f99"),
}

# (css kind, dark hex) -> token. Kind is bg, fg, bd, or sel.
RULES: dict[tuple[str, str], str] = {}


def _bind(token: str, kinds: tuple[str, ...]) -> None:
    dark = PALETTE[token][0]
    for kind in kinds:
        key = (kind, dark)
        previous = RULES.get(key)
        if previous is not None and previous != token:
            raise RuntimeError(f"{kind} {dark} is both {previous} and {token}")
        RULES[key] = token


_bind("bg", ("bg",))
_bind("bg_status", ("bg",))
_bind("bg_stage", ("bg",))
_bind("bg_pressed", ("bg",))
_bind("bg_pressed_2", ("bg",))
_bind("sunken", ("bg",))
_bind("panel", ("bg",))
_bind("panel_alt", ("bg",))
_bind("raised", ("bg",))
_bind("input", ("bg",))
_bind("input_read", ("bg",))
_bind("menu", ("bg",))
_bind("button", ("bg",))
_bind("button_border", ("bd",))
_bind("button_hover", ("bg",))
_bind("button_pressed", ("bg",))
_bind("button_disabled", ("bg",))
_bind("button_disabled_border", ("bd",))
_bind("hover", ("bg",))
_bind("hover_input", ("bg",))
_bind("hover_tool", ("bg",))
_bind("hover_strong", ("bg",))
_bind("hover_soft", ("bg",))
_bind("selected", ("bg",))
_bind("toast", ("bg",))
_bind("toast_border", ("bd",))
_bind("chip", ("bg",))
_bind("text", ("fg",))
_bind("text_menu", ("fg",))
_bind("text_body", ("fg",))
_bind("text_soft", ("fg",))
_bind("text_bright", ("fg",))
_bind("text_strong", ("fg", "bd"))
_bind("text_readout", ("fg",))
_bind("text_toast", ("fg",))
_bind("text_cal", ("fg",))
_bind("text_dim", ("fg",))
_bind("text_secondary", ("fg",))
_bind("text_muted", ("fg",))
_bind("text_faint", ("fg",))
_bind("text_check", ("fg",))
_bind("text_disabled", ("fg",))
_bind("text_disabled_2", ("fg",))
_bind("text_toast_dim", ("fg",))
_bind("border", ("bd",))
_bind("border_bar", ("bd",))
_bind("line", ("bg", "bd"))
_bind("border_input", ("bd",))
_bind("border_menu", ("bd",))
_bind("border_popup", ("bd",))
_bind("border_line", ("bg", "bd"))
_bind("border_btn", ("bd",))
_bind("border_play", ("bd",))
_bind("border_hover", ("bd",))
_bind("border_disabled", ("bd",))
_bind("border_loop", ("bd",))
_bind("border_indicator", ("bd",))
_bind("groove", ("bg",))
_bind("groove_fill", ("bg",))
_bind("selection", ("sel",))
_bind("selection_menu", ("sel",))
_bind("selection_list", ("sel",))
_bind("checked_bg", ("bg",))
_bind("checked_border", ("bd",))
_bind("loop_on", ("bg",))
_bind("accent", ("bg", "bd"))
_bind("accent_bright", ("bd",))
_bind("teal", ("bg", "bd"))
_bind("loop_hover_border", ("bd",))
_bind("assistant_bg", ("bg",))
_bind("assistant_sunken", ("bg",))
_bind("assistant_card", ("bg",))
_bind("assistant_input", ("bg",))
_bind("assistant_button", ("bg",))
_bind("assistant_hover", ("bg",))
_bind("assistant_primary_hover", ("bg",))
_bind("assistant_line", ("bg", "bd"))
_bind("assistant_border", ("bd",))
_bind("assistant_border_2", ("bd",))
_bind("assistant_selection", ("sel",))
_bind("assistant_text", ("fg",))
_bind("assistant_muted", ("fg",))
_bind("assistant_faint", ("fg",))
_bind("assistant_think", ("fg",))
_bind("assistant_disabled", ("fg",))
_bind("assistant_link", ("fg",))
_bind("assistant_link_2", ("fg",))
_bind("assistant_focus", ("bd",))
_bind("chip_ok", ("fg",))
_bind("chip_ok_border", ("bd",))
_bind("chip_warn", ("fg",))
_bind("chip_warn_border", ("bd",))
_bind("chip_bad", ("fg",))
_bind("chip_bad_border", ("bd",))


def name() -> str:
    return _NAME


def is_light() -> bool:
    return _NAME == "light"


def use(scheme: str) -> str:
    global _NAME
    _NAME = "light" if scheme == "light" else "dark"
    return _NAME


_PREFERENCE = "system"


def preference() -> str:
    return _PREFERENCE


def set_preference(value: str) -> str:
    """system follows the OS. light and dark stay put until the user changes them."""
    global _PREFERENCE
    _PREFERENCE = value if value in {"system", "light", "dark"} else "system"
    if _PREFERENCE == "light":
        return use("light")
    if _PREFERENCE == "dark":
        return use("dark")
    return sync_system()


def sync_system() -> str:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is None:
        return use("dark")
    scheme = app.styleHints().colorScheme()
    return use("light" if scheme == Qt.ColorScheme.Light else "dark")


def apply_palette(app) -> None:
    """Scrollbars and other unstyled controls read the palette, not only the sheet."""
    from PySide6.QtGui import QPalette

    if app is None:
        return
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, qcolor("bg"))
    palette.setColor(QPalette.ColorRole.WindowText, qcolor("text"))
    palette.setColor(QPalette.ColorRole.Base, qcolor("panel"))
    palette.setColor(QPalette.ColorRole.AlternateBase, qcolor("sunken"))
    palette.setColor(QPalette.ColorRole.Text, qcolor("text"))
    palette.setColor(QPalette.ColorRole.Button, qcolor("button"))
    palette.setColor(QPalette.ColorRole.ButtonText, qcolor("text"))
    palette.setColor(QPalette.ColorRole.Mid, qcolor("groove"))
    palette.setColor(QPalette.ColorRole.Dark, qcolor("border"))
    palette.setColor(QPalette.ColorRole.Highlight, qcolor("selection"))
    palette.setColor(QPalette.ColorRole.HighlightedText, qcolor("text"))
    app.setPalette(palette)


def color(token: str, scheme: str | None = None) -> str:
    dark, light = PALETTE[token]
    chosen = _NAME if scheme is None else scheme
    return light if chosen == "light" else dark


def qcolor(token: str):
    from PySide6.QtGui import QColor

    return QColor(color(token))


def assistant_message_css() -> str:
    return (
        f"body {{ color: {color('assistant_text')}; background: transparent; font-size: 15px; }}"
        f"a {{ color: {color('assistant_link')}; }}"
        f"code {{ background: {color('assistant_button')}; color: {color('assistant_text')}; }}"
    )


def assistant_thinking_css() -> str:
    return (
        f"body {{ color: {color('assistant_think')}; background: transparent; font-size: 13px; }}"
    )


def stylesheet(scheme: str | None = None) -> str:
    chosen = _NAME if scheme is None else ("light" if scheme == "light" else "dark")
    text = _QSS.read_text(encoding="utf-8")
    for token in PALETTE:
        text = text.replace("{{" + token + "}}", color(token, chosen))
    return text


def tokenize_qss(text: str) -> str:
    """Replace chrome hex colors with {{token}}. Leaves data colors alone."""
    import re

    prop = re.compile(
        r"(background(?:-color)?|color|selection-background-color|selection-color|"
        r"border(?:-top|-bottom|-left|-right|-color)?|gridline-color)"
        r"(\s*:\s*)([^;{}]+)"
    )
    hex_color = re.compile(r"#[0-9a-fA-F]{3,8}")

    def kind_of(prop_name: str) -> str:
        if prop_name.startswith("background"):
            return "bg"
        if prop_name in {"color", "selection-color"}:
            return "fg"
        if prop_name.startswith("selection"):
            return "sel"
        return "bd"

    def repl_prop(match: re.Match[str]) -> str:
        kind = kind_of(match.group(1))

        def repl_hex(hex_match: re.Match[str]) -> str:
            token = RULES.get((kind, hex_match.group(0).lower()))
            if token is None:
                return hex_match.group(0)
            return "{{" + token + "}}"

        return match.group(1) + match.group(2) + hex_color.sub(repl_hex, match.group(3))

    return prop.sub(repl_prop, text)
