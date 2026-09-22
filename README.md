# Faculty CV Intelligence System

Streamlit application for faculty CV/resume extraction and verification.

## Run locally

```bash
pip install -r requirements_cloud.txt
streamlit run app_cloud.py
```

## Streamlit Community Cloud

Set the app entrypoint to `app_cloud.py`.

If AI verification is enabled, add `OPENAI_API_KEY` under
Streamlit Cloud **Settings → Secrets**. Do not commit API keys to GitHub.

The web version uploads PDF resumes through the browser instead of relying
on a local Windows resume folder.
