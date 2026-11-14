import json
import io
from pathlib import Path
from copy import deepcopy

import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ProBad Dataset Reviewer",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ProBad Dataset Reviewer")
st.caption("Review, validate, and annotate Bengali proverb data.")


# ============================================================
# Helpers
# ============================================================

def get_nested(data: dict, *keys, default=None):
    """Safely retrieve a nested dictionary value."""
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def set_nested(data: dict, keys: tuple, value):
    """Create nested dictionaries if necessary and set a value."""
    current = data

    for key in keys[:-1]:
        if not isinstance(current.get(key), dict):
            current[key] = {}

        current = current[key]

    current[keys[-1]] = value


def display_value(value):
    """Convert missing or empty values into a readable placeholder."""
    if value is None or value == "":
        return "—"

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )

    return str(value)


def load_jsonl(file_content: str):
    """
    Parse JSONL line by line.

    Returns:
        records: Valid JSON objects.
        errors: Parsing errors with original line numbers.
    """
    records = []
    errors = []

    for line_number, line in enumerate(
        file_content.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)

            if not isinstance(record, dict):
                errors.append(
                    f"Line {line_number}: Expected a JSON object."
                )
                continue

            # Internal field used to preserve the original line number.
            records.append({
                "_source_line": line_number,
                "data": record,
            })

        except json.JSONDecodeError as exc:
            errors.append(
                f"Line {line_number}: {exc.msg} "
                f"(column {exc.colno})"
            )

    return records, errors


def get_record_id(record):
    return record.get("item_id", "unknown")


def get_proverb(record):
    return get_nested(record, "proverb", default={}) or {}


def get_review_status(record):
    return get_nested(
        record,
        "meta",
        "review_status",
        default="Pending",
    ) or "Pending"


def get_meaning(record, language):
    return get_nested(
        record,
        "proverb",
        "meaning",
        language,
        default="",
    )


def render_dialogue(turns):
    """Render a list of dialogue turns."""
    if not isinstance(turns, list) or not turns:
        st.info("No dialogue available.")
        return

    for turn in turns:
        if not isinstance(turn, dict):
            continue

        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")

        with st.container(border=True):
            st.markdown(f"**{speaker}**")
            st.write(text)


def render_options(options):
    """Display interpretation options."""
    if not isinstance(options, dict) or not options:
        st.info("No interpretation options available.")
        return

    labels = {
        "correct": "✅ Correct interpretation",
        "literal": "📖 Literal interpretation",
        "plausible": "⚠️ Plausible distractor",
    }

    for key, label in labels.items():
        if key in options:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.write(options[key])

    # Display any additional options not covered above.
    for key, value in options.items():
        if key not in labels:
            with st.container(border=True):
                st.markdown(f"**{key.title()}**")
                st.write(value)


def render_metadata(meta):
    """Display metadata in a readable table."""
    if not isinstance(meta, dict):
        st.info("No metadata available.")
        return

    rows = []

    for key, value in meta.items():
        if key == "annotator2":
            continue

        rows.append({
            "Field": key,
            "Value": display_value(value),
        })

    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Sidebar: input
# ============================================================

st.sidebar.header("1. Load dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload JSONL file",
    type=["jsonl", "json", "txt"],
    help="One JSON object per line.",
)

local_path = st.sidebar.text_input(
    "Or enter a local file path",
    placeholder="./data/probad.jsonl",
)

load_clicked = st.sidebar.button(
    "Load dataset",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Session state
# ============================================================

if "records" not in st.session_state:
    st.session_state.records = []

if "load_errors" not in st.session_state:
    st.session_state.load_errors = []

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "probad.jsonl"


# ============================================================
# Load dataset
# ============================================================

if load_clicked:
    file_content = None
    dataset_name = "probad.jsonl"

    if uploaded_file is not None:
        file_content = uploaded_file.getvalue().decode(
            "utf-8-sig"
        )
        dataset_name = uploaded_file.name

    elif local_path.strip():
        path = Path(local_path.strip()).expanduser()

        try:
            file_content = path.read_text(encoding="utf-8-sig")
            dataset_name = path.name

        except (OSError, UnicodeError) as exc:
            st.sidebar.error(f"Could not read file: {exc}")

    else:
        st.sidebar.warning(
            "Upload a file or enter a local file path."
        )

    if file_content is not None:
        records, errors = load_jsonl(file_content)

        # Reset the working dataset when loading a new file.
        st.session_state.records = records
        st.session_state.load_errors = errors
        st.session_state.current_index = 0
        st.session_state.dataset_name = dataset_name

        st.session_state.pop("review_form", None)

        st.sidebar.success(
            f"Loaded {len(records)} valid records."
        )


# ============================================================
# Dataset overview
# ============================================================

records = st.session_state.records

if not records:
    st.info(
        "Upload your ProBad JSONL file or enter its local path "
        "in the sidebar, then click **Load dataset**."
    )

    st.code(
        "streamlit run review_app.py",
        language="bash",
    )

    st.stop()


if st.session_state.load_errors:
    with st.expander(
        f"⚠️ JSON parsing errors "
        f"({len(st.session_state.load_errors)})",
        expanded=True,
    ):
        st.warning(
            "Invalid lines were skipped. They will not appear "
            "in the exported dataset."
        )

        for error in st.session_state.load_errors:
            st.text(error)


# ============================================================
# Dataset statistics
# ============================================================

total_records = len(records)

reviewed_count = sum(
    get_review_status(item["data"]) == "Reviewed"
    for item in records
)

pending_count = total_records - reviewed_count

col1, col2, col3, col4 = st.columns(4)

col1.metric("Valid records", total_records)
col2.metric("Reviewed", reviewed_count)
col3.metric("Pending", pending_count)
col4.metric(
    "JSON errors",
    len(st.session_state.load_errors),
)

st.divider()


# ============================================================
# Filters
# ============================================================

st.subheader("2. Find a proverb")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    search_text = st.text_input(
        "Search",
        placeholder="ID, Bengali, romanisation, or meaning",
    )

with filter_col2:
    split_options = sorted({
        str(
            get_nested(
                item["data"],
                "meta",
                "split",
                default="Unknown",
            )
        )
        for item in records
    })

    selected_split = st.selectbox(
        "Dataset split",
        ["All"] + split_options,
    )

with filter_col3:
    selected_status = st.selectbox(
        "Review status",
        ["All", "Pending", "Reviewed"],
    )


filtered_indices = []

for index, item in enumerate(records):
    record = item["data"]
    proverb = get_proverb(record)

    # Search across the most useful text fields.
    searchable_text = " ".join([
        str(get_record_id(record)),
        str(proverb.get("bengali", "")),
        str(proverb.get("romanised", "")),
        str(proverb.get("respelled", "")),
        str(get_meaning(record, "en")),
        str(get_meaning(record, "bn")),
    ]).casefold()

    if search_text.casefold() not in searchable_text:
        continue

    split = str(
        get_nested(
            record,
            "meta",
            "split",
            default="Unknown",
        )
    )

    if selected_split != "All" and split != selected_split:
        continue

    status = get_review_status(record)

    if selected_status != "All" and status != selected_status:
        continue

    filtered_indices.append(index)


if not filtered_indices:
    st.warning("No records match the selected filters.")
    st.stop()


# ============================================================
# Navigation
# ============================================================

if st.session_state.current_index not in filtered_indices:
    st.session_state.current_index = filtered_indices[0]

current_position = filtered_indices.index(
    st.session_state.current_index
)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(
    [1, 1, 3, 2]
)

with nav_col1:
    if st.button(
        "← Previous",
        use_container_width=True,
        disabled=current_position == 0,
    ):
        st.session_state.current_index = filtered_indices[
            current_position - 1
        ]
        st.rerun()

with nav_col2:
    if st.button(
        "Next →",
        use_container_width=True,
        disabled=current_position == len(filtered_indices) - 1,
    ):
        st.session_state.current_index = filtered_indices[
            current_position + 1
        ]
        st.rerun()

with nav_col3:
    selected_record_id = st.selectbox(
        "Jump to record",
        options=filtered_indices,
        index=current_position,
        format_func=lambda i: (
            f"{get_record_id(records[i]['data'])} "
            f"— {get_proverb(records[i]['data']).get('bengali', '')}"
        ),
    )

    if selected_record_id != st.session_state.current_index:
        st.session_state.current_index = selected_record_id
        st.rerun()

with nav_col4:
    st.caption(
        f"Record {current_position + 1} "
        f"of {len(filtered_indices)} filtered"
    )


# ============================================================
# Current record
# ============================================================

current_index = st.session_state.current_index
record = records[current_index]["data"]
proverb = get_proverb(record)

item_id = get_record_id(record)

st.divider()

title_col1, title_col2 = st.columns([4, 1])

with title_col1:
    st.header(f"Proverb: {item_id}")

with title_col2:
    status = get_review_status(record)

    if status == "Reviewed":
        st.success("Reviewed")
    else:
        st.warning("Pending")


# ============================================================
# Proverb details
# ============================================================

st.subheader("3. Proverb details")

st.markdown(
    f"<h2 style='font-size: 2rem;'>{proverb.get('bengali', '—')}</h2>",
    unsafe_allow_html=True,
)

st.caption(
    f"Proverb ID: {proverb.get('proverb_id', '—')}"
)

detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.markdown("**Romanised**")
    st.write(display_value(proverb.get("romanised")))

    st.markdown("**Respelled**")
    st.write(display_value(proverb.get("respelled")))

    st.markdown("**Literal gloss**")
    st.write(display_value(proverb.get("literal_gloss")))

with detail_col2:
    st.markdown("**English meaning**")
    st.write(display_value(get_meaning(record, "en")))

    st.markdown("**Bengali meaning**")
    st.write(display_value(get_meaning(record, "bn")))

    st.markdown("**Region**")
    st.write(display_value(proverb.get("region")))

    st.markdown("**Figurative**")
    st.write(display_value(proverb.get("figurative")))


with st.expander("Cultural notes and false friends", expanded=True):
    st.markdown("**Cultural note**")
    st.write(display_value(proverb.get("cultural_note")))

    st.markdown("**False friends**")
    false_friends = proverb.get("false_friends", [])

    if false_friends:
        for item in false_friends:
            st.markdown(f"- {item}")
    else:
        st.write("None recorded.")


# ============================================================
# Memorisation
# ============================================================

st.subheader("4. Memorisation")

memorisation = record.get("memorisation", {}) or {}

mem_col1, mem_col2 = st.columns(2)

with mem_col1:
    st.markdown("**Bengali**")

    bn_mem = memorisation.get("bengali", {}) or {}

    st.write(
        f"Prefix: {bn_mem.get('prefix', '—')}"
    )
    st.write(
        f"Target: {bn_mem.get('target', '—')}"
    )

with mem_col2:
    st.markdown("**Romanised**")

    roman_mem = memorisation.get("romanised", {}) or {}

    st.write(
        f"Prefix: {roman_mem.get('prefix', '—')}"
    )
    st.write(
        f"Target: {roman_mem.get('target', '—')}"
    )


# ============================================================
# Dialogue review
# ============================================================

st.subheader("5. Dialogue review")

language_col1, language_col2 = st.columns(2)

for column, language, title in [
    (language_col1, "en", "🇬🇧 English"),
    (language_col2, "bn", "🇧🇩 Bengali"),
]:
    with column:
        st.markdown(f"### {title}")

        language_data = record.get(language, {}) or {}

        with st.container(border=True):
            st.markdown("**Conversation**")
            render_dialogue(language_data.get("turns", []))

            st.markdown("**Proverb speaker**")
            st.write(
                display_value(
                    language_data.get("proverb_speaker")
                )
            )

        st.markdown("**Interpretation options**")
        render_options(language_data.get("options", {}))


# ============================================================
# Maps and metadata
# ============================================================

st.subheader("6. Maps and metadata")

maps_col, metadata_col = st.columns(2)

with maps_col:
    st.markdown("**Maps**")

    maps = record.get("maps", {}) or {}

    st.write(
        f"In maps: {display_value(maps.get('in_maps'))}"
    )

    st.write(
        f"Maps reference: {display_value(maps.get('maps_ref'))}"
    )

with metadata_col:
    st.markdown("**Metadata**")
    render_metadata(record.get("meta", {}))


# ============================================================
# Annotation
# ============================================================

st.divider()

st.subheader("7. Annotation")

annotator2 = get_nested(
    record,
    "meta",
    "annotator2",
    default={},
) or {}

# Existing values are used as defaults.
# A form prevents intermediate keystrokes from being saved.
with st.form(key=f"annotation_{current_index}"):

    status_options = ["Pending", "Reviewed"]

    current_status = get_review_status(record)

    status_index = (
        status_options.index(current_status)
        if current_status in status_options
        else 0
    )

    review_status = st.selectbox(
        "Review status",
        status_options,
        index=status_index,
    )

    label = st.text_input(
        "Annotator label",
        value=str(annotator2.get("label") or ""),
        help="Enter the label assigned by the second annotator.",
    )

    agreement_options = [
        "Not assessed",
        "Agreed",
        "Disagreed",
    ]

    existing_agreed = annotator2.get("agreed")

    if existing_agreed is True:
        default_agreement = "Agreed"
    elif existing_agreed is False:
        default_agreement = "Disagreed"
    else:
        default_agreement = "Not assessed"

    agreement_index = agreement_options.index(
        default_agreement
    )

    agreement = st.selectbox(
        "Agreement with original annotation",
        agreement_options,
        index=agreement_index,
    )

    comment = st.text_area(
        "Reviewer comments",
        value=str(annotator2.get("comment") or ""),
        height=120,
        placeholder="Record any corrections, ambiguities, or issues.",
    )

    save_clicked = st.form_submit_button(
        "Save annotation",
        type="primary",
        use_container_width=True,
    )

    if save_clicked:
        # Update only the annotation and review status.
        set_nested(
            record,
            ("meta", "annotator2", "label"),
            label.strip() or None,
        )

        agreed_value = {
            "Not assessed": None,
            "Agreed": True,
            "Disagreed": False,
        }[agreement]

        set_nested(
            record,
            ("meta", "annotator2", "agreed"),
            agreed_value,
        )

        set_nested(
            record,
            ("meta", "annotator2", "comment"),
            comment.strip() or None,
        )

        set_nested(
            record,
            ("meta", "review_status"),
            review_status,
        )

        st.session_state.records[current_index]["data"] = record

        st.success(
            f"Saved annotation for {item_id}. "
            "Download the dataset to persist your changes."
        )

        st.rerun()


# ============================================================
# Raw JSON inspection
# ============================================================

with st.expander("Inspect raw JSON"):
    st.json(record, expanded=False)


# ============================================================
# Export
# ============================================================

st.divider()

st.subheader("8. Export reviewed dataset")

st.write(
    "Export all successfully loaded records, including those "
    "that have not been reviewed. Existing fields are preserved."
)

export_col1, export_col2 = st.columns(2)

with export_col1:
    output = io.StringIO()

    for item in records:
        output.write(
            json.dumps(
                item["data"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        output.write("\n")

    st.download_button(
        label="⬇️ Download reviewed JSONL",
        data=output.getvalue().encode("utf-8"),
        file_name=f"reviewed_{st.session_state.dataset_name}",
        mime="application/x-ndjson",
        use_container_width=True,
    )

with export_col2:
    st.download_button(
        label="⬇️ Download validation errors",
        data="\n".join(
            st.session_state.load_errors
        ).encode("utf-8"),
        file_name="jsonl_validation_errors.txt",
        mime="text/plain",
        disabled=not st.session_state.load_errors,
        use_container_width=True,
    )