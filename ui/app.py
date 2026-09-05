"""Streamlit interface for the pneumonia detection API."""

import os

import requests
import streamlit as st

API_URL = os.getenv("PNEUMONIA_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Détection de pneumonie",
    page_icon="🫁",
    layout="centered",
)

st.title("Détection de pneumonie")
st.caption("Prototype académique d'aide à la décision sur radiographies thoraciques")

st.warning(
    "Cette application ne constitue pas un dispositif médical et ne remplace pas "
    "l'interprétation d'un radiologue."
)

uploaded_file = st.file_uploader(
    "Importer une radiographie thoracique",
    type=["jpg", "jpeg", "png"],
    help="Formats JPEG ou PNG, 10 Mo maximum.",
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Radiographie importée", width="stretch")
    include_gradcam = st.checkbox("Afficher l'explication Grad-CAM", value=True)

    if st.button("Analyser la radiographie", type="primary", width="stretch"):
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    params={"include_gradcam": str(include_gradcam).lower()},
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"L'API de prédiction est indisponible : {exc}")
            else:
                probability = result["pneumonia_probability"]
                if result["prediction"] == "PNEUMONIA":
                    st.error("Résultat : suspicion de pneumonie")
                else:
                    st.success("Résultat : aspect classé normal")

                col1, col2 = st.columns(2)
                col1.metric("Probabilité pneumonie", f"{probability:.1%}")
                col2.metric("Seuil de décision", f"{result['threshold']:.0%}")
                st.progress(probability)

                if result.get("gradcam_base64"):
                    st.subheader("Zones influençant la prédiction")
                    st.image(
                        result["gradcam_base64"],
                        caption="Carte Grad-CAM - les zones chaudes influencent le modèle",
                        width="stretch",
                    )
                with st.expander("Informations techniques"):
                    st.json(
                        {
                            "modèle": result["model"],
                            "identifiant": result["request_id"],
                            "probabilité normale": result["normal_probability"],
                        }
                    )
                st.caption(result["warning"])
