import streamlit as st
import pandas as pd
import numpy as np
import io
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# -------------------------------------
# Konfiguracja strony
# -------------------------------------
st.set_page_config(page_title="🔎 Ukryta kanibalizacja (pełny raport)", layout="wide")

st.title("🔎 Analiza ukrytej kanibalizacji + scalanie briefów z wytycznymi + finalny raport")

# API Key
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# Parametry
threshold = st.sidebar.slider("Próg podobieństwa (cosine similarity)", 0.70, 0.95, 0.80, 0.01)

# Upload pliku
uploaded_file = st.file_uploader("Wgraj plik briefy_pelne.xlsx (z poprzedniego etapu)", type=["xlsx"])


# -------------------------------------
# Funkcja do embeddingów
# -------------------------------------
def get_embeddings(texts, client, model="text-embedding-3-large"):
    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data])


# -------------------------------------
# Funkcje do scalania tytułów i wytycznych
# -------------------------------------
def merge_title(titles, client):
    """Scalanie wielu tytułów w jeden dominujący"""
    text = " | ".join(set([t for t in titles if isinstance(t, str) and t.strip() != ""]))
    if not text:
        return ""
    prompt = f"""
    Na podstawie poniższych tytułów scalonych artykułów stwórz JEDEN nowy tytuł,
    który najlepiej oddaje ich wspólny sens.
    Tytuł powinien być zwięzły, naturalny i SEO-friendly (maksymalnie 70 znaków).
    Bez cudzysłowów, bez znaków specjalnych.

    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


def merge_guidelines(guidelines, client):
    """Scalanie wielu wytycznych w jedne spójne"""
    text = " | ".join(set([g for g in guidelines if isinstance(g, str) and g.strip() != ""]))
    if not text:
        return ""
    prompt = f"""
    Na podstawie poniższych wytycznych scal je w jedne spójne i całościowe
    wskazówki dla autora artykułu. Napisz zwięźle w 1-2 zdaniach.
    Bez cudzysłowów, bez znaków specjalnych.

    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80
    )
    return response.choices[0].message.content.strip()


# -------------------------------------
# Logika główna
# -------------------------------------
if uploaded_file and OPENAI_API_KEY:
    progress = st.progress(0)
    status = st.empty()

    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Podgląd danych wejściowych")
    st.dataframe(df.head())

    client = OpenAI(api_key=OPENAI_API_KEY)

    # --- krok 1: wybór tylko pojedynczych ---
    singles = df[df["status"] == "pojedynczy"].reset_index(drop=True)
    titles = singles["tytul"].astype(str).tolist()
    ids = singles["cluster_ids"].astype(str).tolist()
    n = len(singles)

    status.text(f"📥 Załadowano {n} pojedynczych klastrów do analizy...")
    progress.progress(20)

    # --- krok 2: embeddingi ---
    status.text("🧠 Tworzenie embeddingów dla tytułów...")
    embeddings = get_embeddings(titles, client, "text-embedding-3-large")
    progress.progress(50)

    # --- krok 3: macierz podobieństw ---
    status.text("🔍 Obliczanie podobieństw między tytułami...")
    sim_matrix = cosine_similarity(embeddings)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                pairs.append({
                    "cluster_id_1": ids[i],
                    "title_1": titles[i],
                    "cluster_id_2": ids[j],
                    "title_2": titles[j],
                    "similarity": round(float(sim), 4)
                })
    progress.progress(70)

    # --- krok 4: wyniki par ---
    if pairs:
        results_df = pd.DataFrame(pairs).sort_values(by="similarity", ascending=False).reset_index(drop=True)
        st.subheader("📑 Podejrzane pary (ukryta kanibalizacja)")
        st.dataframe(results_df)

        # --- krok 5: grupowanie par w klastry ---
        G = nx.Graph()
        for _, row in results_df.iterrows():
            G.add_edge(row["cluster_id_1"], row["cluster_id_2"])

        groups = list(nx.connected_components(G))
        group_data = []
        for i, g in enumerate(groups, start=1):
            group_titles = singles[singles["cluster_ids"].isin(g)]["tytul"].tolist()
            group_data.append({
                "group_id": i,
                "titles": group_titles,
                "count": len(group_titles)
            })
        grouped_df = pd.DataFrame(group_data)
        st.subheader("📦 Zgrupowane klastry kanibalizacji")
        st.dataframe(grouped_df)

        # --- krok 6: scalone briefy z nowym tytułem i wytycznymi ---
        briefs = []
        for i, g in enumerate(groups, start=1):
            group_df = singles[singles["cluster_ids"].isin(g)]

            main_phrase = group_df["main_phrase"].iloc[0]
            intent = group_df["intencja"].mode()[0] if not group_df["intencja"].mode().empty else group_df["intencja"].iloc[0]

            merged_phrases = ", ".join(set(group_df["frazy"].dropna()))
            merged_ids = ", ".join(map(str, group_df["cluster_ids"].tolist()))

            merged_title = merge_title(group_df["tytul"].dropna().tolist(), client)
            merged_guidelines = merge_guidelines(group_df["wytyczne"].dropna().tolist(), client) if "wytyczne" in group_df.columns else ""

            briefs.append({
                "status": "scalone_2etap",
                "group_id": i,
                "cluster_ids": merged_ids,
                "main_phrase": main_phrase,
                "intencja": intent,
                "frazy": merged_phrases,
                "tytul": merged_title,
                "wytyczne": merged_guidelines
            })

        briefs_df = pd.DataFrame(briefs)
        st.subheader("📝 Briefy dla scalonych klastrów (2 etap, z jednym tytułem i wytycznymi)")
        st.dataframe(briefs_df)

        # --- krok 7: przygotowanie finalnego zestawienia ---
        scalone_ids = set(",".join(briefs_df["cluster_ids"]).split(", "))
        df_filtered = df[~df["cluster_ids"].astype(str).isin(scalone_ids)]

        final_df = pd.concat([df_filtered, briefs_df], ignore_index=True)

        # --- eksport ---
        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Briefy_oryginalne", index=False)  # wszystkie oryginalne (pojedyncze + scalone_1etap)
            results_df.to_excel(writer, sheet_name="Ukryta_kanibalizacja", index=False)
            grouped_df.to_excel(writer, sheet_name="Grupy", index=False)
            briefs_df.to_excel(writer, sheet_name="Briefy_scalone_2etap", index=False)
            final_df.to_excel(writer, sheet_name="Finalne_briefy", index=False)

        xlsx_buffer.seek(0)

        st.download_button(
            label="📥 Pobierz pełny raport (Finalne briefy)",
            data=xlsx_buffer,
            file_name="pelny_raport_briefow.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        status.text("✅ Zakończono!")
        progress.progress(100)

    else:
        st.success("✅ Nie wykryto podejrzanych par powyżej progu podobieństwa.")

