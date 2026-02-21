"""
Monkey-patch for Marker's TableProcessor.split_combined_rows to handle
financial statement tables where the table recognition model merges all
transaction rows into a single row.

The core problem: Marker's built-in split logic requires all columns to have
the same number of text lines. Financial statements break this because:
  - Date column: 1 line per transaction
  - Description column: 1-2 lines per transaction (merchant + city)
  - Reward Points column: sparse (many transactions have no points)
  - Amount column: 1 line per transaction

Our fix uses the Date column as the row-count anchor and distributes
other columns' lines to the correct transaction row using bbox y-positions.
"""

import re
from copy import deepcopy

# Date pattern: DD/MM/YYYY with optional HH:MM:SS
DATE_PATTERN = re.compile(r'^\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2}:\d{2})?\s*$')

# Amount pattern: Indian format numbers with optional "Cr" suffix
AMOUNT_PATTERN = re.compile(
    r'^\s*[\d,]+\.\d{2}(?:\s*Cr)?\s*$'
)


def _get_text(line):
    """Extract text string from a text_line entry (dict or object)."""
    if isinstance(line, dict):
        return line.get("text", "")
    if hasattr(line, "text"):
        return line.text
    return str(line)


def _get_y(line):
    """Extract y-top position from a text_line entry. Returns None if no bbox."""
    if isinstance(line, dict):
        bbox = line.get("bbox", None)
        if bbox and len(bbox) >= 2:
            return bbox[1]
    if hasattr(line, "bbox") and line.bbox and len(line.bbox) >= 2:
        return line.bbox[1]
    return None


def _is_date_line(text):
    """Check if a text line looks like a transaction date."""
    return bool(DATE_PATTERN.match(text.strip()))


def _is_amount_line(text):
    """Check if a text line looks like a transaction amount."""
    return bool(AMOUNT_PATTERN.match(text.strip()))


def _identify_anchor_column(row_cells):
    """
    Find the best anchor column — the one where every non-empty line is a date
    or an amount. Prefer the date column (usually col 0).
    Returns (col_index_in_row, anchor_count, anchor_type) or None.
    """
    best = None
    for ci, cell in enumerate(row_cells):
        if not cell.text_lines or not isinstance(cell.text_lines, list):
            continue
        lines = cell.text_lines
        if len(lines) < 2:
            continue
        date_count = sum(1 for l in lines if _is_date_line(_get_text(l)))
        amt_count = sum(1 for l in lines if _is_amount_line(_get_text(l)))
        # Date column: most lines are dates
        if date_count >= len(lines) * 0.7:
            if best is None or best[2] == "amount":
                best = (ci, date_count, "date")
        # Amount column: most lines are amounts
        elif amt_count >= len(lines) * 0.7:
            if best is None:
                best = (ci, amt_count, "amount")
    return best


def _build_row_bands(anchor_lines):
    """
    Build vertical bands from anchor lines (dates/amounts).
    Each band is (y_start, y_end, line_index).
    Lines whose y-center falls within a band belong to that row.
    """
    ys = []
    for i, line in enumerate(anchor_lines):
        y = _get_y(line)
        if y is not None:
            ys.append((y, i))
    if not ys:
        return None

    ys.sort(key=lambda x: x[0])
    bands = []
    for idx, (y, line_idx) in enumerate(ys):
        if idx == 0:
            y_start = 0  # from top of cell
        else:
            y_start = (ys[idx - 1][0] + y) / 2  # midpoint with previous
        if idx == len(ys) - 1:
            y_end = float('inf')  # to bottom of cell
        else:
            y_end = (y + ys[idx + 1][0]) / 2  # midpoint with next
        bands.append((y_start, y_end, line_idx))
    return bands


def _assign_lines_to_bands(lines, bands, n_rows):
    """
    Assign text lines to row bands based on their y-position.
    Falls back to equal distribution when bbox data is unavailable (OCR tables).
    Returns a list of n_rows lists.
    """
    result = [[] for _ in range(n_rows)]
    if not lines or not bands:
        return result

    # Check if any lines have bbox data
    has_bbox = any(_get_y(line) is not None for line in lines)

    if not has_bbox:
        # OCR fallback: distribute lines as evenly as possible
        return _distribute_evenly(lines, n_rows)

    for line in lines:
        y = _get_y(line)
        if y is None:
            # No bbox on this specific line — assign to nearest populated row
            # or first row as fallback
            for i in range(n_rows):
                if result[i]:
                    result[i].append(line)
                    break
            else:
                result[0].append(line)
            continue

        # Find which band this line's y-center belongs to
        assigned = False
        for y_start, y_end, row_idx in bands:
            if y_start <= y < y_end and row_idx < n_rows:
                result[row_idx].append(line)
                assigned = True
                break
        if not assigned:
            # Assign to the closest band
            min_dist = float('inf')
            best_row = 0
            for y_start, y_end, row_idx in bands:
                band_center = (y_start + y_end) / 2 if y_end != float('inf') else y_start + 20
                dist = abs(y - band_center)
                if dist < min_dist and row_idx < n_rows:
                    min_dist = dist
                    best_row = row_idx
            result[best_row].append(line)

    return result


def _distribute_evenly(lines, n_rows):
    """
    Distribute lines across n_rows as evenly as possible.
    Used as fallback when bbox data is unavailable (OCR tables).
    """
    result = [[] for _ in range(n_rows)]
    if not lines:
        return result
    if len(lines) <= n_rows:
        for i, line in enumerate(lines):
            result[i].append(line)
    else:
        per_row = len(lines) / n_rows
        for i in range(n_rows):
            start = int(i * per_row)
            end = int((i + 1) * per_row)
            result[i] = lines[start:end]
    return result


def patched_split_combined_rows(self, tables):
    """
    Enhanced split_combined_rows that handles financial statement tables
    where columns have unequal line counts.

    Uses bbox y-positions from the anchor column (dates) to define row bands,
    then assigns all other columns' lines to the correct band by y-position.
    """
    from surya.table_rec.schema import TableCell as SuryaTableCell

    for table in tables:
        if len(table.cells) == 0:
            continue

        unique_rows = sorted(list(set([c.row_id for c in table.cells])))
        new_cells = []
        shift_up = 0
        max_cell_id = max([c.cell_id for c in table.cells])
        new_cell_count = 0
        any_split = False

        for row in unique_rows:
            row_cells = deepcopy([c for c in table.cells if c.row_id == row])
            line_lens = [
                len(c.text_lines) if isinstance(c.text_lines, list) else 0
                for c in row_cells
            ]
            max_lines = max(line_lens) if line_lens else 0

            # Skip rows that don't need splitting
            if max_lines <= 1 or len(row_cells) <= 1:
                for cell in row_cells:
                    cell.row_id += shift_up
                    new_cells.append(cell)
                continue

            # Check if this looks like a collapsed financial transaction row
            anchor = _identify_anchor_column(row_cells)
            if anchor is None:
                for cell in row_cells:
                    cell.row_id += shift_up
                    new_cells.append(cell)
                continue

            anchor_ci, n_rows, anchor_type = anchor
            if n_rows < 2:
                for cell in row_cells:
                    cell.row_id += shift_up
                    new_cells.append(cell)
                continue

            # Build row bands from anchor column's y-positions
            anchor_lines = row_cells[anchor_ci].text_lines
            bands = _build_row_bands(anchor_lines)
            # bands may be None for OCR tables where lines lack bbox data

            # Split this row into n_rows individual rows
            any_split = True
            for ci, cell in enumerate(row_cells):
                tl = cell.text_lines if isinstance(cell.text_lines, list) else []

                if ci == anchor_ci:
                    # Anchor column: 1:1 mapping
                    distributed = [[tl[i]] if i < len(tl) else [] for i in range(n_rows)]
                elif len(tl) == 0:
                    distributed = [[] for _ in range(n_rows)]
                elif bands is not None:
                    # Use bbox y-positions to assign lines to bands
                    distributed = _assign_lines_to_bands(tl, bands, n_rows)
                else:
                    # No bbox data (OCR tables) — distribute evenly
                    distributed = _distribute_evenly(tl, n_rows)

                split_height = (cell.bbox[3] - cell.bbox[1]) / max(n_rows, 1)
                for i in range(n_rows):
                    current_bbox = [
                        cell.bbox[0],
                        cell.bbox[1] + i * split_height,
                        cell.bbox[2],
                        cell.bbox[1] + (i + 1) * split_height,
                    ]
                    cell_id = max_cell_id + new_cell_count
                    new_cells.append(
                        SuryaTableCell(
                            polygon=current_bbox,
                            text_lines=distributed[i] if distributed[i] else None,
                            rowspan=1,
                            colspan=cell.colspan,
                            row_id=cell.row_id + shift_up + i,
                            col_id=cell.col_id,
                            is_header=cell.is_header and i == 0,
                            within_row_id=cell.within_row_id,
                            cell_id=cell_id,
                        )
                    )
                    new_cell_count += 1

            shift_up += n_rows - 1

        if any_split and len(new_cells) > len(table.cells):
            table.cells = new_cells


def patch_table_processor():
    """Apply the monkey-patch to Marker's TableProcessor."""
    try:
        from marker.processors.table import TableProcessor
        TableProcessor.split_combined_rows = patched_split_combined_rows
        return True
    except ImportError:
        return False
