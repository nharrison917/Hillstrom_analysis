# Streamlit Gotchas — Hillstrom Project

Reference for recurring issues encountered building `08_app.py`.

---

## 1. Dollar signs trigger LaTeX rendering

**Symptom:** Text like `$0.63` renders as broken math, e.g. `*0.63*` or blank.

**Why:** `st.caption()`, `st.markdown()`, and most other text elements render markdown,
and Streamlit treats `$...$` as LaTeX math delimiters.

**Fix:** Escape with `\$` wherever a literal dollar sign appears in rendered text.

```python
# Bad
st.caption("Mean CATE: **$0.63** | Mens: **$0.72**")

# Good
st.caption(r"Mean CATE: **\$0.63** | Mens: **\$0.72**")
```

Use a raw string (`r"..."`) or double-escape (`"\\$"`) — either works.
Apply this in `st.caption()`, `st.markdown()`, `st.info()`, `st.success()`, etc.
Does NOT apply to `st.code()` (literal text block, no markdown processing).

---

## 2. `use_container_width` deprecated — affects charts AND dataframes

**Symptom:** Console warning:
```
Please replace `use_container_width` with `width`.
`use_container_width` will be removed after 2025-12-31.
For `use_container_width=True`, use `width='stretch'`.
```

**Why:** Streamlit deprecated the boolean `use_container_width` parameter across
`st.plotly_chart`, `st.dataframe`, and other display elements.

**Fix:** Replace everywhere — both chart and dataframe calls.

```python
# Bad (deprecated)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df, hide_index=True, use_container_width=True)

# Good
st.plotly_chart(fig, width="stretch")
st.dataframe(df, hide_index=True, width="stretch")
```

Note: `use_container_width=False` maps to `width='content'`.

---

## 3. `st.components.v1.html` deprecated — use `st.html()`

**Symptom:** Console warning:
```
`st.components.v1.html` will be removed after 2026-06-01.
```

**Why:** The old components API is being phased out.

**Fix:** `st.html()` is NOT a drop-in replacement for `components.html()` when the HTML
is a full standalone document (i.e. contains `<html>`, `<head>`, `<script>` tags, such as
a Plotly export). `components.html()` renders in an iframe with its own JS context;
`st.html()` renders inline and does not execute scripts reliably.

- For **simple HTML snippets** (no scripts): `st.html()` works fine.
- For **simple HTML snippets** (no scripts): `st.html()` works fine.
- For **full Plotly/standalone HTML files**: use `st.iframe()` — it replaces `components.html()`
  with the same iframe sandbox. API is identical: `height` and `scrolling` params work the same.

```python
# Works for simple snippets only
st.html("<p>Some <strong>text</strong></p>")

# Correct replacement for full Plotly HTML exports
st.iframe(html_content, height=500)  # scrolling param not supported
```

Migration path summary:
- `components.html(content)` → `st.iframe(content)` (full HTML documents, scripts)
- `components.html(snippet)` → `st.html(snippet)` (simple markup, no scripts)

---

## 4. `st.cache_data` hides file changes until cache is cleared

**Symptom:** Editing an output file (e.g. `outputs/07_policy_rules.txt`) and refreshing
the browser shows the old content. The fix appears to have no effect.

**Why:** `@st.cache_data` on a file-reading function caches the return value in memory.
The browser refresh re-runs the script but hits the cache, not disk.

**Fix:** Clear the cache explicitly — one of:
- In the browser: hamburger menu (top-right) → "Clear cache"
- Restart the Streamlit server: `Ctrl+C` then `streamlit run 08_app.py`

This applies to any function decorated with `@st.cache_data` or `@st.cache_resource`
that reads from files that may change between runs.

---

## 5. Deprecation warnings affect ALL call sites, not just charts

When a deprecation warning fires repeatedly (one per page load, not per element),
it typically means multiple elements use the deprecated API — not just the first one found.
Search the whole file before assuming a single fix resolves it.

```bash
grep -n "use_container_width" 08_app.py
```
