from pathlib import Path
from typing import Any, Iterable, Optional


def tab_render(tab_heads: list[str], tab_rows: Iterable[Iterable[Any]]) -> str:
    tab_data = [[str(tab_cell) for tab_cell in tab_row] for tab_row in tab_rows]
    tab_widths = [len(tab_head) for tab_head in tab_heads]

    for tab_row in tab_data:
        if len(tab_row) != len(tab_heads):
            raise ValueError("Table row length does not match header length")
        for tab_pos, tab_cell in enumerate(tab_row):
            tab_widths[tab_pos] = max(tab_widths[tab_pos], len(tab_cell))

    tab_border = "+" + "+".join("-" * (tab_width + 2) for tab_width in tab_widths) + "+"

    def tab_line(tab_row: Iterable[str]) -> str:
        return (
            "|"
            + "|".join(
                f" {tab_cell:<{tab_widths[tab_pos]}} "
                for tab_pos, tab_cell in enumerate(tab_row)
            )
            + "|"
        )

    tab_lines = [tab_border, tab_line(tab_heads), tab_border]
    tab_lines.extend(tab_line(tab_row) for tab_row in tab_data)
    tab_lines.append(tab_border)
    return "\n".join(tab_lines)


def tab_output(tab_text: str, tab_path: Optional[str]) -> None:
    print(tab_text)
    if tab_path:
        tab_file = Path(tab_path)
        tab_file.parent.mkdir(parents=True, exist_ok=True)
        tab_file.write_text(tab_text + "\n", encoding="ascii", errors="replace")
