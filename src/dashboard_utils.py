from __future__ import annotations

from pathlib import Path
import io
import pandas as pd
import streamlit as st

CACHE_TTL = 600

@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.is_file():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read CSV: `{path.name}` → {e}")
            return None
    return None

@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def _list_images(
    folder: Path, patterns=(".png", ".jpg", ".jpeg", ".svg", ".webp")
) -> list[Path]:
    if not folder.exists():
        return []
    files: set[Path] = set()
    for ext in patterns:
        suffix = ext.lstrip(".")
        ci_pattern = "*." + "".join(
            f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in suffix
        )
        files.update(folder.glob(ci_pattern))
    return sorted(files)

@st.cache_data(show_spinner=False)
def _read_file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()

def _show_images_grid(
    image_paths: list[Path], cols: int = 2, caption_from_name: bool = True, max_per_page: int = 6
):
    if not image_paths:
        st.info("No figures found yet. Generate them via your EDA/training scripts.")
        return

    total_pages = (len(image_paths) + max_per_page - 1) // max_per_page
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=f"page_{hash(str(image_paths))}")
        start_idx = (page - 1) * max_per_page
        end_idx = min(start_idx + max_per_page, len(image_paths))
        current_paths = image_paths[start_idx:end_idx]
        st.caption(f"Showing {start_idx+1}-{end_idx} of {len(image_paths)} images")
    else:
        current_paths = image_paths

    for i in range(0, len(current_paths), cols):
        row_paths = current_paths[i : i + cols]
        cols_objs = st.columns(len(row_paths))
        for c, p in zip(cols_objs, row_paths):
            with c:
                try:
                    img_bytes = _read_file_bytes(str(p))
                    if p.suffix.lower() == ".svg":
                        st.image(io.BytesIO(img_bytes))
                    else:
                        st.image(img_bytes)
                    if caption_from_name:
                        st.caption(p.stem.replace("_", " ").title())
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=p.name,
                        mime="image/svg+xml" if p.suffix.lower() == ".svg" else None,
                        key=f"dl_{p.name}_{i}",
                    )
                except Exception as e:
                    st.warning(f"Failed to display `{p.name}` → {e}")

def _show_table(csv_path: Path, title: str):
    df = _safe_read_csv(csv_path)
    if df is None:
        st.info(f"`{csv_path.name}` not available. Create it in the `tables/` folder.")
        return
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=csv_path.name,
        mime="text/csv",
        key=f"dl_{csv_path.name}",
    )

def clear_caches() -> None:
    _safe_read_csv.clear()
    _list_images.clear()
    _read_file_bytes.clear()
