import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from dash import Input, Output, State, MATCH, ALL, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dcc import Download, send_data_frame
import json
from dash import no_update
import dash

# =========================
# Dados
# =========================
CSV_PATH_V5 = "comparacao_medidas_17092025.csv"
CSV_PATH_V4 = "comparacao_medidas_v5.csv"
# CSV_PATH_V4 = "comparacao_medidas_v4.csv"

# v5 = base principal (todo o app usa v5 e parser atual)
df = pd.read_csv(CSV_PATH_V5)

# v4 = usado apenas para o gráfico de % OK (LLM v4)
try:
    df_v4 = pd.read_csv(CSV_PATH_V4)
except Exception:
    # Se não existir, mantém vazio
    df_v4 = pd.DataFrame(columns=["ClassificationType", "Medidas_llm"])

# Corrigir booleanos se necessário (v5)
for col in ["Atingiu_minimo", "Atingiu_mediana", "Atingiu_maximo"]:
    if col in df.columns and df[col].dtype != bool:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

# Lista de classes (com "Todas") — baseada no v5
class_list = sorted(df["ClassificationType"].dropna().unique().tolist())
class_options = [{"label": "Todas", "value": "__ALL__"}] + [{"label": c, "value": c} for c in class_list]

# Mapeamento do modo -> (flag, esperado, label)
MODE_TO_FLAG = {
    "min":    ("Atingiu_minimo",  "min_expected_measurement",   "Mínimo"),
    "median": ("Atingiu_mediana", "median_expected_measurement","Mediana"),
    "max":    ("Atingiu_maximo",  "max_expected_measurement",   "Máximo"),
}

def _subset_by_classification(df_in: pd.DataFrame, selected_class: str) -> pd.DataFrame:
    if selected_class and selected_class != "__ALL__": 
        return df_in[df_in["ClassificationType"] == selected_class].copy()
    return df_in.copy()

def kpi_llm_parser(dfin: pd.DataFrame):
    s_llm = pd.to_numeric(dfin["Medidas_llm"], errors="coerce").fillna(0).sum()
    s_par = pd.to_numeric(dfin["Medidas_parser"], errors="coerce").fillna(0).sum()
    total = s_llm + s_par
    pct_llm = (s_llm / total * 100) if total > 0 else 0
    pct_par = (s_par / total * 100) if total > 0 else 0
    return int(s_llm), int(s_par), int(total), round(pct_llm, 1), round(pct_par, 1)

def agg_by_classification(dfin: pd.DataFrame):
    g = dfin.groupby("ClassificationType")[["Medidas_llm","Medidas_parser"]].sum().reset_index()
    g["Total"] = g["Medidas_llm"] + g["Medidas_parser"]
    g["LLM_div_Parser"] = g["Medidas_llm"] / g["Medidas_parser"].replace(0, np.nan)
    return g.sort_values("Total", ascending=False)

def pct_by_classification_split_v5(dfin: pd.DataFrame, mode: str):
    """Mantém versão antiga (v5 apenas) se precisar em outro lugar."""
    _, exp_col, _ = MODE_TO_FLAG[mode]
    dff = dfin.copy()
    dff["Medidas_llm"] = pd.to_numeric(dff["Medidas_llm"], errors="coerce").fillna(0)
    dff["Medidas_parser"] = pd.to_numeric(dff["Medidas_parser"], errors="coerce").fillna(0)
    dff[exp_col] = pd.to_numeric(dff[exp_col], errors="coerce").fillna(0)
    dff["ok_llm"] = dff["Medidas_llm"]   >= dff[exp_col]
    dff["ok_parser"] = dff["Medidas_parser"] >= dff[exp_col]
    g = dff.groupby("ClassificationType").agg(
        pct_llm=("ok_llm", "mean"),
        pct_parser=("ok_parser", "mean"),
    ).reset_index()
    g["pct_llm"] = (g["pct_llm"] * 100).round(1)
    g["pct_parser"] = (g["pct_parser"] * 100).round(1)
    return g

def pct_by_classification_split_versioned(df_v5: pd.DataFrame, df_v4: pd.DataFrame, mode: str):
    """
    Retorna um DF por classe com %OK de:
      - LLM v4 (usando Medidas_llm do v4 comparado ao esperado do v5 para o mode)
      - LLM v5
      - Parser v5
    Filtra para classes onde o esperado (no v5) > 0 no mode.
    """
    _, exp_col, _label = MODE_TO_FLAG[mode]

    # Mapa de esperado por classe (v5)
    exp_map = (
        df_v5.groupby("ClassificationType")[exp_col]
        .max()  # se houver valores repetidos por classe, usa o maior
        .rename("Expected")
        .reset_index()
    )

    # ---- v5: calcula % ok por classe para LLM e Parser
    d5 = df_v5.copy()
    d5["Medidas_llm"] = pd.to_numeric(d5["Medidas_llm"], errors="coerce").fillna(0)
    d5["Medidas_parser"] = pd.to_numeric(d5["Medidas_parser"], errors="coerce").fillna(0)
    d5 = d5.merge(exp_map, on="ClassificationType", how="left")
    d5["Expected"] = pd.to_numeric(d5["Expected"], errors="coerce").fillna(0)

    d5["ok_llm_v5"] = d5["Medidas_llm"] >= d5["Expected"]
    d5["ok_par_v5"] = d5["Medidas_parser"] >= d5["Expected"]

    g5 = d5.groupby("ClassificationType").agg(
        pct_llm_v5=("ok_llm_v5", "mean"),
        pct_par_v5=("ok_par_v5", "mean"),
        Expected=("Expected", "max"),
    ).reset_index()

    # ---- v4: junta esperado por classe e calcula % ok LLM
    if not df_v4.empty and "Medidas_llm" in df_v4.columns and "ClassificationType" in df_v4.columns:
        d4 = df_v4[["ClassificationType", "Medidas_llm"]].copy()
        d4["Medidas_llm"] = pd.to_numeric(d4["Medidas_llm"], errors="coerce").fillna(0)
        d4 = d4.merge(exp_map, on="ClassificationType", how="left")
        d4["Expected"] = pd.to_numeric(d4["Expected"], errors="coerce").fillna(0)
        d4["ok_llm_v4"] = d4["Medidas_llm"] >= d4["Expected"]
        g4 = d4.groupby("ClassificationType").agg(pct_llm_v4=("ok_llm_v4", "mean")).reset_index()
    else:
        g4 = pd.DataFrame(columns=["ClassificationType", "pct_llm_v4"])

    # ---- merge v5 + v4 por classe
    g = g5.merge(g4, on="ClassificationType", how="left")
    g["pct_llm_v4"] = g["pct_llm_v4"].fillna(0.0)

    # filtra classes com Expected > 0 (no v5)
    g = g[g["Expected"] > 0].copy()

    # formata %
    for c in ["pct_llm_v4", "pct_llm_v5", "pct_par_v5"]:
        g[c] = (g[c] * 100).round(1)

    # ordena por "destaque" do v5 (média entre llm e parser v5)
    g["rank_key"] = (g["pct_llm_v5"] + g["pct_par_v5"]).fillna(0)
    g = g.sort_values("rank_key", ascending=False).drop(columns=["rank_key"])
    return g

# =========================
# Estilos
# =========================
TABLE_BASE_COLS = [
    {"name": "EHR_ID", "id": "EHR_ID"},
    {"name": "Patient_ID", "id": "Patient_ID"},
    {"name": "Total", "id": "Total"},
    {"name": "Esperado", "id": "Esperado"},
    {"name": "LLM", "id": "Medidas_llm"},
    {"name": "Parser", "id": "Medidas_parser"},
]
STYLE_TABLE = {"height": "360px", "overflowY": "auto"}
STYLE_DATA = {"whiteSpace": "normal", "height": "auto", "fontSize": "13px"}
STYLE_HEADER = {"fontWeight": "600", "backgroundColor": "#f8f9fa"}

# =========================
# Layout helpers
# =========================
def make_mode_section(mode: str):
    _, _, label = MODE_TO_FLAG[mode]
    return dbc.Card(
        dbc.CardBody([
            html.Hr(),
            html.H4(f"Limiar: {label}", className="mb-3"),

            # <<< Gráfico de % OK (agora versionado) >>>
            dcc.Loading(dcc.Graph(id={"type": "bar_pct", "mode": mode}), type="dot"),

            dbc.Row([
                dbc.Col([
                    html.H5("Falhas"),
                    html.Div("EHRs que NÃO atingiram o limiar", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id={"type": "tbl_fail", "mode": mode},
                        columns=TABLE_BASE_COLS,
                        data=[],
                        page_size=10,
                        row_selectable="single",
                        style_table=STYLE_TABLE,
                        style_data=STYLE_DATA,
                        style_header=STYLE_HEADER,
                        sort_action="native",
                    ),
                    dbc.Button(
                        "Baixar CSV (Falhas)",
                        id={"type": "btn_download_fail", "mode": mode},
                        color="danger", outline=True, size="sm", className="mt-2"
                    ),
                    Download(id={"type": "download_fail", "mode": mode}),
                ], md=6),
                dbc.Col([
                    html.H5("Sucessos"),
                    html.Div("EHRs que atingiram o limiar", className="text-muted mb-2"),
                    dash_table.DataTable(
                        id={"type": "tbl_success", "mode": mode},
                        columns=TABLE_BASE_COLS,
                        data=[],
                        page_size=10,
                        row_selectable="single",
                        style_table=STYLE_TABLE,
                        style_data=STYLE_DATA,
                        style_header=STYLE_HEADER,
                        sort_action="native",
                    ),
                    dbc.Button(
                        "Baixar CSV (Sucessos)",
                        id={"type": "btn_download_success", "mode": mode},
                        color="success", outline=True, size="sm", className="mt-2"
                    ),
                    Download(id={"type": "download_success", "mode": mode}),
                ], md=6),
            ], className="g-3"),

            html.Div("Dica: clique em um ponto do gráfico, selecione uma linha das tabelas acima "
                     "ou digite um EHR_ID para abrir o painel global.",
                     className="mt-2 text-muted")
        ]),
        className="mb-3 shadow-sm"
    )

# --- helpers para o painel lado a lado ---
def _safe_get(d, k, default="-"):
    try:
        v = d.get(k, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v
    except Exception:
        return default

def _clip(text, n=120):
    if text is None:
        return ""
    s = str(text)
    return (s[:n] + "…") if len(s) > n else s

def _summarize_row(entry: dict, kind: str) -> str:
    if not isinstance(entry, dict) or not entry:
        return ""
    k = kind.lower()
    if k.startswith("meas"):
        name  = _clip(entry.get("MeasurementName", "-"))
        val   = _clip(entry.get("StringValue", "-"))
        side  = _clip(entry.get("Side", "-"))
        etype = _clip(entry.get("extractionType", "-"))
        return f"{name} • {val} • Side={side} • Type={etype}"
    if k.startswith("presc"):
        med   = _clip(entry.get("MedicationName", "-"))
        side  = _clip(entry.get("Side", "-"))
        freq  = _clip(entry.get("Frequency", entry.get("FrequencyInstructions", "-")))
        dur   = _clip(entry.get("Duration", "-"))
        dose  = _clip(entry.get("Dosage", entry.get("MedicationDosage", "-")))
        unit  = _clip(entry.get("MedicationDosageUnit", ""))
        route = _clip(entry.get("RouteOfAdministration", "-"))
        dose_txt = f"{dose}{unit}" if unit and dose != "-" else f"{dose}"
        return f"{med} • Dose={dose_txt} • Freq={freq} • Dur={dur} • Side={side} • Route={route}"
    disp  = _clip(entry.get("DisplayName", "-"))
    side  = _clip(entry.get("Side", "-"))
    span  = _clip(entry.get("PeriodSpan", "-"))
    pdate = _clip(entry.get("ProcedureDate", "-"))
    etype = _clip(entry.get("extractionType", "-"))
    return f"{disp} • Side={side} • Period={span} • Date={pdate} • Type={etype}"

def _json_tt(entry: dict) -> str:
    try:
        return f"```json\n{json.dumps(entry or {}, indent=2, ensure_ascii=False)}\n```"
    except Exception:
        return "```json\n{}\n```"

def _side_by_side_table(llm_entries, parser_entries, kind="Measurements"):
    llm_entries = llm_entries if isinstance(llm_entries, list) else []
    parser_entries = parser_entries if isinstance(parser_entries, list) else []
    n = max(len(llm_entries), len(parser_entries))

    data = []
    tooltip_data = []

    for i in range(n):
        l = llm_entries[i] if i < len(llm_entries) else {}
        p = parser_entries[i] if i < len(parser_entries) else {}

        llm_side = str(_safe_get(l, "Side", "-"))
        par_side = str(_safe_get(p, "Side", "-"))

        llm_summary = _summarize_row(l, kind)
        par_summary = _summarize_row(p, kind)

        data.append({
            "LLM": llm_summary,
            "Parser": par_summary,
            "LLM_Side": llm_side,
            "Parser_Side": par_side
        })

        tooltip_data.append({
            "LLM": {"value": _json_tt(l), "type": "markdown"},
            "Parser": {"value": _json_tt(p), "type": "markdown"},
            "LLM_Side": llm_side,
            "Parser_Side": par_side,
        })

    table_id = f"tbl_side_by_side_{kind.lower()}"

    return dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": "LLM", "id": "LLM", "presentation": "markdown"},
            {"name": "LLM Side", "id": "LLM_Side"},
            {"name": "Parser", "id": "Parser", "presentation": "markdown"},
            {"name": "Parser Side", "id": "Parser_Side"},
        ],
        data=data,
        tooltip_data=tooltip_data,
        tooltip_duration=None,
        sort_action="native",
        page_size=10,
        style_table={"maxHeight": "50vh", "overflowY": "auto"},
        style_cell={"whiteSpace": "normal", "height": "auto", "fontSize": "13px"},
        style_header={"fontWeight": "600", "backgroundColor": "#f8f9fa"},
        css=[
            {
                "selector": ".dash-table-tooltip",
                "rule": (
                    "max-width: 900px;"
                    "width: 900px;"
                    "max-height: 70vh;"
                    "overflow-y: auto;"
                    "white-space: pre-wrap;"
                    "font-family: monospace;"
                    "font-size: 12px;"
                    "z-index: 9999;"
                ),
            },
            {
                "selector": ".dash-table-tooltip .markdown",
                "rule": "white-space: pre-wrap;",
            },
        ],
    )

# =========================
# App / Layout
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "EHR Coverage Dashboard"

navbar = dbc.Navbar(
    dbc.Container([dbc.NavbarBrand("EHR Coverage Dashboard", className="ms-2")]),
    color="primary", dark=True, sticky="top"
)

header_block = html.Div([
    html.Br(),
    html.H2("Comparativo LLM vs Parser", className="mb-2"),
])

compare_card = dbc.Card(
    dbc.CardBody([
        # KPIs
        dbc.Row([
            dbc.Col(dbc.Card(
                dbc.CardBody([html.Div("Total Medidas", className="text-muted"), html.H3(id="kpi_total")]),
                className="border-0 shadow-sm"
            ), md=3),
            dbc.Col(dbc.Card(
                dbc.CardBody([html.Div("LLM (soma / %)", className="text-muted"), html.H3(id="kpi_llm")]),
                className="border-0 shadow-sm"
            ), md=3),
            dbc.Col(dbc.Card(
                dbc.CardBody([html.Div("Parser (soma / %)", className="text-muted"), html.H3(id="kpi_parser")]),
                className="border-0 shadow-sm"
            ), md=3),
            dbc.Col(
                dbc.Checklist(
                    id="chk_norm_percent",
                    options=[{"label": " Mostrar barras como %", "value": "pct"}],
                    value=[],
                    switch=True,
                ), md=3, className="d-flex align-items-center justify-content-end"
            )
        ], className="g-3 mb-3"),

        # Gráficos comparativos (SEM versionar — permanece como antes)
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="bar_stack_class")), md=7),
            dbc.Col(dcc.Loading(dcc.Graph(id="line_ratio_class")), md=5),
        ], className="g-3"),

        dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id="scatter_ehr")), md=12)], className="g-3 mt-3"),

        # Painel Global de EHR
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.Div("Buscar EHR específico", className="fw-bold mb-2"),
                dbc.InputGroup([
                    dcc.Input(
                        id="ehr_search_input",
                        placeholder="Digite o EHR_ID e pressione Enter ou clique em Mostrar",
                        type="number",
                        debounce=True,
                        style={"width": "100%"}
                    ),
                    dbc.Button("Mostrar", id="ehr_search_btn", n_clicks=0, color="primary")
                ], className="mb-2"),
                html.Small("Dica: clique em um ponto do gráfico, selecione uma linha das tabelas, ou use o campo acima."),
                html.Div(
                    id="ehr_detail_global",
                    style={
                        "whiteSpace": "pre-wrap",
                        "fontFamily": "monospace",
                        "fontSize": "13px",
                        "lineHeight": "1.35",
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid #e9ecef",
                        "borderRadius": "6px",
                        "padding": "12px",
                        "maxHeight": "60vh",
                        "overflowY": "auto",
                        "wordBreak": "break-word",
                        "overflowWrap": "anywhere",
                        "overflowX": "hidden",
                    }
                )
            ], md=12),
        ], className="g-3 mt-1"),
    ]),
    className="mb-3 shadow-sm"
)

mode_checklist = dbc.Checklist(
    id="mode_checklist",
    options=[
        {"label": " Mínimo",  "value": "min"},
        {"label": " Mediana", "value": "median"},
        {"label": " Máximo",  "value": "max"},
    ],
    value=["min"],
    inline=True,
    switch=False,
    className="mb-2",
)

controls = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Classificação", className="fw-bold"),
                dcc.Dropdown(id="class_dropdown", options=class_options, value="__ALL__", clearable=False),
            ], md=6),
            dbc.Col([
                html.Label("Opções", className="fw-bold"),
                dbc.Checklist(
                    id="chk_norm_percent",
                    options=[{"label": " Mostrar barras como %", "value": "pct"}],
                    value=[],
                    switch=True,
                ),
            ], md=6),
        ], className="g-3"),
        html.Div([html.Span("Limiar para % OK (multi seleção): ", className="me-2 fw-bold"), mode_checklist])
    ]),
    className="mb-3 shadow-sm"
)

sections_container = html.Div(id="sections_container")

# ---- Modal de JSON ----
json_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Detalhe (JSON)")),
    dbc.ModalBody(html.Pre(id="json_modal_body",
                           style={"whiteSpace":"pre-wrap","maxHeight":"70vh","overflowY":"auto"})),
], id="json_modal", is_open=False, size="lg", scrollable=True)

# ---- Stores para listas completas (usados pelo modal) ----
stores_block = html.Div([
    dcc.Store(id="store_meas_llm"),
    dcc.Store(id="store_meas_parser"),
    dcc.Store(id="store_rx_llm"),
    dcc.Store(id="store_rx_parser"),
    dcc.Store(id="store_proc_llm"),
    dcc.Store(id="store_proc_parser"),
])

app.layout = dbc.Container([
    navbar,
    header_block,
    controls,
    compare_card,
    sections_container,
    json_modal,
    stores_block,
], fluid=True)

# =========================
# Callbacks
# =========================

# KPIs
@app.callback(
    Output("kpi_total", "children"),
    Output("kpi_llm", "children"),
    Output("kpi_parser", "children"),
    Input("class_dropdown", "value"),
)
def update_kpis(selected_class):
    dff = _subset_by_classification(df, selected_class)
    s_llm, s_par, total, pct_llm, pct_par = kpi_llm_parser(dff)
    fmt = lambda x: f"{int(x):,}".replace(",", ".")
    return (fmt(total), f"{fmt(s_llm)} / {pct_llm:.1f}%", f"{fmt(s_par)} / {pct_par:.1f}%")

# Barras empilhadas (LLM vs Parser) por classe (SEM versionar; segue como estava)
@app.callback(
    Output("bar_stack_class", "figure"),
    Input("class_dropdown", "value"),
    Input("chk_norm_percent", "value"),
)
def update_bar_stack(selected_class, chk):
    dff = _subset_by_classification(df, "__ALL__")
    g = agg_by_classification(dff)
    show_pct = "pct" in (chk or [])
    if show_pct:
        g["LLM_pct"] = (g["Medidas_llm"] / g["Total"].replace(0, np.nan)) * 100
        g["Parser_pct"] = (g["Medidas_parser"] / g["Total"].replace(0, np.nan)) * 100
        fig = px.bar(g, x="ClassificationType", y=["LLM_pct","Parser_pct"],
                     labels={"value":"% dentro da classe", "ClassificationType":"Classificação", "variable":""},
                     barmode="stack")
        fig.update_yaxes(ticksuffix="%")
    else:
        fig = px.bar(g, x="ClassificationType", y=["Medidas_llm","Medidas_parser"],
                     labels={"value":"Medidas (soma)", "ClassificationType":"Classificação", "variable":""},
                     barmode="stack")
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), legend_title_text="")
    return fig

# Razão LLM/Parser por classe (segue v5)
@app.callback(
    Output("line_ratio_class", "figure"),
    Input("class_dropdown", "value"),
)
def update_ratio(selected_class):
    dff = _subset_by_classification(df, "__ALL__")
    g = agg_by_classification(dff)
    g["LLM_div_Parser"] = g["LLM_div_Parser"].replace([np.inf, -np.inf], np.nan).fillna(0)
    fig = px.line(g, x="ClassificationType", y="LLM_div_Parser", markers=True,
                  labels={"LLM_div_Parser": "Razão LLM / Parser", "ClassificationType": "Classificação"})
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    return fig

# Scatter por EHR (segue v5)
@app.callback(
    Output("scatter_ehr", "figure"),
    Input("class_dropdown", "value"),
)
def update_scatter(selected_class):
    dff = _subset_by_classification(df, selected_class)
    dff = dff.copy()
    dff["Medidas_llm"] = pd.to_numeric(dff["Medidas_llm"], errors="coerce")
    dff["Medidas_parser"] = pd.to_numeric(dff["Medidas_parser"], errors="coerce")
    dff = dff.dropna(subset=["Medidas_llm","Medidas_parser"])
    fig = px.scatter(
        dff, x="Medidas_parser", y="Medidas_llm",
        hover_data=["PartnerID", "EHR_ID","Patient_ID","ClassificationType"],
        custom_data=["EHR_ID"],
        labels={"Medidas_parser":"Medidas Parser (por EHR/classe)",
                "Medidas_llm":"Medidas LLM (por EHR/classe)"}
    )
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    return fig

# Construção dinâmica das seções (multi-limiar)
@app.callback(
    Output("sections_container", "children"),
    Input("mode_checklist", "value"),
)
def build_sections(modes):
    if not modes:
        return dbc.Alert("Selecione pelo menos um limiar (Mínimo/Mediana/Máximo).", color="warning")
    order = ["min", "median", "max"]
    return [make_mode_section(m) for m in order if m in modes]

# ==== Barras % OK por modo (VERSIONADO: LLM v4, LLM v5, Parser v5) ====
@app.callback(
    Output({"type": "bar_pct", "mode": MATCH}, "figure"),
    Input("mode_checklist", "value"),
    State({"type": "bar_pct", "mode": MATCH}, "id"),
)
def update_bar_section(modes_selected, this_id):
    mode = this_id["mode"]

    # Dataframe combinado com %OK das três séries, já filtrado por Expected>0 (v5)
    g = pct_by_classification_split_versioned(df, df_v4, mode)

    # Long format
    g_long = g.melt(
        id_vars=["ClassificationType"],
        value_vars=["pct_llm_v4", "pct_llm_v5", "pct_par_v5"],
        var_name="Fonte",
        value_name="% OK"
    )
    g_long["Fonte"] = g_long["Fonte"].map({
        "pct_llm_v4": "LLM v4",
        "pct_llm_v5": "LLM v5",
        "pct_par_v5": "Parser v5"
    })

    fig = px.bar(
        g_long, x="ClassificationType", y="% OK", color="Fonte", barmode="group",
        labels={"ClassificationType": "Classificação"}
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>% OK (%{legendgroup}): %{y:.1f}%<extra></extra>")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_ticksuffix="%", legend_title_text="")
    return fig

# Tabelas de falhas/sucessos por modo (segue v5)
@app.callback(
    Output({"type": "tbl_fail", "mode": MATCH}, "data"),
    Output({"type": "tbl_success", "mode": MATCH}, "data"),
    Input("class_dropdown", "value"),
    State({"type": "tbl_fail", "mode": MATCH}, "id"),
)
def update_tables_section(selected_class, this_id):
    mode = this_id["mode"]
    flag_col, exp_col, _ = MODE_TO_FLAG[mode]
    if selected_class is None:
        return [], []

    dff = _subset_by_classification(df, selected_class).copy()
    dff["Esperado"] = dff[exp_col]

    # Falhas: ordenar pelo maior "gap" (esperado - total), depois total asc
    fails = dff[dff[flag_col] == False].copy()
    fails["gap"] = dff["Esperado"].fillna(0) - dff["Total"].fillna(0)
    fails = fails.sort_values(["gap", "Total"], ascending=[False, True])

    # Sucessos: ordenar por total desc
    succs = dff[dff[flag_col] == True].copy()
    succs = succs.sort_values(["Total"], ascending=False)

    cols = ["EHR_ID", "Patient_ID", "Total", "Esperado", "Medidas_llm", "Medidas_parser"]
    return fails[cols].to_dict("records"), succs[cols].to_dict("records")

# Downloads CSV por seção (segue v5)
@app.callback(
    Output({"type": "download_fail", "mode": MATCH}, "data"),
    Input({"type": "btn_download_fail", "mode": MATCH}, "n_clicks"),
    State({"type": "tbl_fail", "mode": MATCH}, "data"),
    State({"type": "btn_download_fail", "mode": MATCH}, "id"),
    prevent_initial_call=True
)
def download_fail_section(n, data, this_id):
    mode = this_id["mode"]
    df_out = pd.DataFrame(data)
    return send_data_frame(df_out.to_csv, f"falhas_{mode}.csv", index=False)

@app.callback(
    Output({"type": "download_success", "mode": MATCH}, "data"),
    Input({"type": "btn_download_success", "mode": MATCH}, "n_clicks"),
    State({"type": "tbl_success", "mode": MATCH}, "data"),
    State({"type": "btn_download_success", "mode": MATCH}, "id"),
    prevent_initial_call=True
)
def download_success_section(n, data, this_id):
    mode = this_id["mode"]
    df_out = pd.DataFrame(data)
    return send_data_frame(df_out.to_csv, f"sucessos_{mode}.csv", index=False)

# -------------------------
# DETALHE GLOBAL do EHR + Stores (segue v5)
# -------------------------
@app.callback(
    Output("ehr_detail_global", "children"),
    Output("store_meas_llm", "data"),
    Output("store_meas_parser", "data"),
    Output("store_rx_llm", "data"),
    Output("store_rx_parser", "data"),
    Output("store_proc_llm", "data"),
    Output("store_proc_parser", "data"),
    Input("scatter_ehr", "clickData"),
    Input("ehr_search_btn", "n_clicks"),
    Input({"type": "tbl_fail", "mode": ALL}, "active_cell"),
    Input({"type": "tbl_success", "mode": ALL}, "active_cell"),
    Input({"type": "tbl_fail", "mode": ALL}, "selected_rows"),
    Input({"type": "tbl_success", "mode": ALL}, "selected_rows"),
    State("ehr_search_input", "value"),
    State({"type": "tbl_fail", "mode": ALL}, "data"),
    State({"type": "tbl_success", "mode": ALL}, "data"),
    State("mode_checklist", "value"),
    prevent_initial_call=True
)
def show_global_ehr(clickData, n_clicks, active_fail, active_succ,
                    sel_fail, sel_succ, ehr_id_input,
                    data_fail_list, data_succ_list, modes_selected):

    trg = ctx.triggered_id
    ehr_id = None

    # 1) Clique no scatter
    if trg == "scatter_ehr" and clickData and clickData.get("points"):
        try:
            ehr_id = clickData["points"][0]["customdata"][0]
        except Exception:
            ehr_id = None

    # 2) Botão "Mostrar"
    elif trg == "ehr_search_btn" and ehr_id_input is not None:
        ehr_id = int(ehr_id_input)

    # 3) Clique/seleção nas tabelas
    elif isinstance(trg, dict) and trg.get("type") in ("tbl_fail", "tbl_success"):
        clicked_mode = trg.get("mode")
        modes_order = [m for m in ["min", "median", "max"] if m in (modes_selected or [])]
        if clicked_mode in modes_order:
            idx = modes_order.index(clicked_mode)
            row_idx = None
            if trg["type"] == "tbl_fail":
                if active_fail and len(active_fail) > idx and active_fail[idx]:
                    row_idx = active_fail[idx].get("row")
                if row_idx is None and sel_fail and len(sel_fail) > idx and sel_fail[idx]:
                    if len(sel_fail[idx]) > 0:
                        row_idx = sel_fail[idx][0]
                if (row_idx is not None) and data_fail_list and len(data_fail_list) > idx:
                    try:
                        ehr_id = data_fail_list[idx][row_idx]["EHR_ID"]
                    except Exception:
                        ehr_id = None
            else:
                if active_succ and len(active_succ) > idx and active_succ[idx]:
                    row_idx = active_succ[idx].get("row")
                if row_idx is None and sel_succ and len(sel_succ) > idx and sel_succ[idx]:
                    if len(sel_succ[idx]) > 0:
                        row_idx = sel_succ[idx][0]
                if (row_idx is not None) and data_succ_list and len(data_succ_list) > idx:
                    try:
                        ehr_id = data_succ_list[idx][row_idx]["EHR_ID"]
                    except Exception:
                        ehr_id = None

    if ehr_id is None:
        return "Nenhum EHR selecionado.", None, None, None, None, None, None

    # Recuperar linhas desse EHR
    ehr_info = df[df["EHR_ID"] == ehr_id]
    if ehr_info.empty:
        return html.Div("EHR não encontrado."), None, None, None, None, None, None

    # Escolher a linha “representativa”
    row = ehr_info.sort_values("Total", ascending=False).iloc[0]
    formatted_text = row.get("FormattedText", "")
    patient_id = row.get("Patient_ID", "")
    ehr_date = row.get("EHR_Date", "")
    partner_id = row.get("PartnerID", "")

    # Derivar "modo ativo" (min/median/max) e rótulo
    order = [m for m in ["min", "median", "max"] if m in (modes_selected or [])]
    active_mode = None
    if isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") in ("tbl_fail", "tbl_success"):
        active_mode = ctx.triggered_id.get("mode")
    elif order:
        active_mode = order[0]
    label = MODE_TO_FLAG[active_mode][2] if active_mode else "-"

    # Helpers numéricos
    def _as_int(x):
        try:
            v = pd.to_numeric(x, errors="coerce")
            if pd.isna(v):
                return 0
            return int(v)
        except Exception:
            return 0

    class_used = row.get("ClassificationType", "-")
    min_exp = _as_int(row.get("min_expected_measurement"))
    med_exp = _as_int(row.get("median_expected_measurement"))
    max_exp = _as_int(row.get("max_expected_measurement"))
    med_llm = _as_int(row.get("Medidas_llm"))
    med_parser = _as_int(row.get("Medidas_parser"))
    total = _as_int(row.get("Total"))

    if active_mode == "min":
        cur_exp = min_exp
    elif active_mode == "median":
        cur_exp = med_exp
    elif active_mode == "max":
        cur_exp = max_exp
    else:
        cur_exp = "-"

    # Cabeçalho
    lines = [
        f"PartnerID: {partner_id} |EHR_ID: {ehr_id} | Patient_ID: {patient_id} | Data: {ehr_date}",
        f"Classificação: {class_used}",
        f"Limiar atual ({label}): {cur_exp}",
        f"Esperados (min/med/max): {min_exp} / {med_exp} / {max_exp}",
        f"Extraído — LLM: {med_llm} | Parser: {med_parser} | Total: {total}",
        "",
        "----- FormattedText -----",
        (formatted_text if isinstance(formatted_text, str) else str(formatted_text))[:200000],
    ]

    # Parse JSON (string -> list[dict])
    def _parse_list(colname):
        series = ehr_info[colname].dropna()
        if series.empty:
            return []
        val = series.iloc[0]
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except:
                return []
        return []

    measurements_llm   = _parse_list("measurements_llm")
    measurements_parser= _parse_list("measurements_parser")
    prescriptions_llm  = _parse_list("prescriptions_llm")
    prescriptions_parser = _parse_list("prescriptions_parser")
    procedures_llm     = _parse_list("procedures_llm")
    procedures_parser  = _parse_list("procedures_parser")

    # Bloco da esquerda (EHR)
    left_block = html.Div([
        html.Pre("\n".join(lines), style={"whiteSpace": "pre-wrap", "fontSize": "14px"})
    ], style={"width": "50%"})
    
    # Bloco da direita (tabelas lado a lado)
    right_block = html.Div([
        html.H4("Extrações (Medidas, Prescrições e Procedimentos)"),

        html.H5("Measurements"),
        _side_by_side_table(measurements_llm, measurements_parser, kind="Measurements"),

        html.Br(),
        html.H5("Prescriptions"),
        _side_by_side_table(prescriptions_llm, prescriptions_parser, kind="Prescriptions"),

        html.Br(),
        html.H5("Procedures"),
        _side_by_side_table(procedures_llm, procedures_parser, kind="Procedures"),
    ], style={"width": "50%"})

    content = html.Div([left_block, right_block], style={"display": "flex", "gap": "30px"})

    # Retornar também os stores para uso no modal
    return (content,
            measurements_llm, measurements_parser,
            prescriptions_llm, prescriptions_parser,
            procedures_llm, procedures_parser)

# -------------------------
# Modal JSON (Measurements/Prescriptions/Procedures)
# -------------------------
@app.callback(
    Output("json_modal","is_open"),
    Output("json_modal_body","children"),
    Input("tbl_side_by_side_measurements","active_cell"),
    Input("tbl_side_by_side_prescriptions","active_cell"),
    Input("tbl_side_by_side_procedures","active_cell"),
    State("store_meas_llm","data"),
    State("store_meas_parser","data"),
    State("store_rx_llm","data"),
    State("store_rx_parser","data"),
    State("store_proc_llm","data"),
    State("store_proc_parser","data"),
    prevent_initial_call=True
)
def open_json_modal(ac_meas, ac_rx, ac_proc,
                    meas_llm, meas_parser, rx_llm, rx_parser, proc_llm, proc_parser):
    # Descobrir qual tabela disparou
    ac = None
    src_table = None
    if ac_meas is not None:
        ac = ac_meas; src_table = "meas"
    elif ac_rx is not None:
        ac = ac_rx;   src_table = "rx"
    elif ac_proc is not None:
        ac = ac_proc; src_table = "proc"
    else:
        return False, dash.no_update

    row = ac.get("row")
    col = ac.get("column_id")  # "LLM" ou "Parser"
    if row is None or col not in ("LLM", "Parser"):
        return False, dash.no_update

    # Selecionar lista correta
    if src_table == "meas":
        lst = meas_llm if col == "LLM" else meas_parser
    elif src_table == "rx":
        lst = rx_llm if col == "LLM" else rx_parser
    else:
        lst = proc_llm if col == "LLM" else proc_parser

    try:
        obj = (lst or [])[row]
    except Exception:
        obj = {}

    body = json.dumps(obj or {}, indent=2, ensure_ascii=False)
    return True, body

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
