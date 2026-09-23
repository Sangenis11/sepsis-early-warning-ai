# =========================================================
# SEPSIS EARLY WARNING AI DASHBOARD
# Prediction of Sepsis Onset Within the Next 6 Hours
# =========================================================

import os
import traceback
import joblib
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


# =========================================================
# CONFIG & MODEL LOADING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "LM6 (Random Forest)": "RandomForest_LM6.joblib",
    "LM12 (Logistic Regression)": "Logistic_LM12.joblib",
    "LM18 (Logistic Regression)": "Logistic_LM18.joblib",
    "LM24 (Logistic Regression)": "Logistic_LM24.joblib"
}

LANDMARK_ORDER = [
    "LM6 (Random Forest)",
    "LM12 (Logistic Regression)",
    "LM18 (Logistic Regression)",
    "LM24 (Logistic Regression)"
]

WINDOW_LABELS = {
    "LM6 (Random Forest)": "6–12h",
    "LM12 (Logistic Regression)": "12–18h",
    "LM18 (Logistic Regression)": "18–24h",
    "LM24 (Logistic Regression)": "24–30h"
}

models = {}
for name, file in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, file)
    if os.path.exists(path):
        try:
            models[name] = joblib.load(path)
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Failed loading {file}: {e}")
            traceback.print_exc()

print("Models available:", list(models.keys()))


# =========================================================
# CLINICAL INPUT VALIDATION
# =========================================================

def validate_inputs(age, hr, spo2, temp, rr, map_ni):
    errors = []
    inputs = {
        "Age": (age, 18, 110),
        "Heart Rate": (hr, 30, 220),
        "SpO₂": (spo2, 50, 100),
        "Temperature": (temp, 30, 43),
        "Respiratory Rate": (rr, 5, 60),
        "MAP": (map_ni, 20, 200)
    }

    for name, (val, min_val, max_val) in inputs.items():
        if val is None:
            errors.append(f"{name} is required.")
        elif not (min_val <= val <= max_val):
            errors.append(f"{name} must be between {min_val} and {max_val}.")

    return errors


# =========================================================
# DASH APP — PROFESSIONAL CLINICAL RESEARCH INTERFACE
# =========================================================

app = dash.Dash(__name__)
app.title = "Sepsis Early Warning AI"

server = app.server

COLORS = {
    "navy": "#12304A", "navy2": "#1B4965", "teal": "#0F766E",
    "teal_light": "#E6F4F1", "blue": "#2563EB", "blue_light": "#EFF6FF",
    "green": "#15803D", "green_light": "#DCFCE7", "amber": "#D97706",
    "amber_light": "#FEF3C7", "red": "#DC2626", "red_light": "#FEE2E2",
    "ink": "#172033", "muted": "#64748B", "border": "#E2E8F0",
    "bg": "#F5F8FA", "white": "#FFFFFF"
}

GLOBAL_STYLE = {
    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "color": COLORS["ink"]
}
CARD = {
    "backgroundColor": COLORS["white"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "16px",
    "boxShadow": "0 4px 18px rgba(15,23,42,.05)"
}
LABEL = {
    "fontSize": "13px", "fontWeight": "600", "color": COLORS["muted"],
    "marginBottom": "7px", "display": "block"
}
INPUT = {
    "width": "100%", "height": "42px",
    "border": f"1px solid {COLORS['border']}", "borderRadius": "9px",
    "padding": "0 12px", "fontSize": "14px", "boxSizing": "border-box",
    "backgroundColor": "#FFFFFF"
}

gender_options = [
    {"label": "Femme", "value": "Female"},
    {"label": "Homme", "value": "Male"}
]
race_options = [
    {"label": "Asian", "value": "Asian"},
    {"label": "Black", "value": "Black"},
    {"label": "Hispanic/Latino", "value": "Hispanic/Latino"},
    {"label": "Native American", "value": "Native American"},
    {"label": "Other / Unknown", "value": "Other/Unknown"},
    {"label": "Pacific Islander", "value": "Pacific Islander"},
    {"label": "White", "value": "White"}
]
binary_options = [
    {"label": "Non", "value": 0}, {"label": "Oui", "value": 1}
]
gcs_options = [
    {"label": "Sévère (≤ 8)", "value": 1},
    {"label": "Modéré (9–12)", "value": 2},
    {"label": "Léger (13–15)", "value": 3}
]
elix_options = [
    {"label": "Faible", "value": 1},
    {"label": "Modéré", "value": 2},
    {"label": "Élevé", "value": 3}
]
default_model = list(models.keys())[0] if models else None

def section_header(num, title, subtitle=""):
    return html.Div([
        html.Div(str(num), style={
            "width":"30px","height":"30px","borderRadius":"9px",
            "backgroundColor":COLORS["teal_light"],"color":COLORS["teal"],
            "display":"flex","alignItems":"center","justifyContent":"center",
            "fontWeight":"800","fontSize":"13px","flexShrink":"0"
        }),
        html.Div([
            html.Div(title, style={"fontSize":"17px","fontWeight":"750","color":COLORS["navy"]}),
            html.Div(subtitle, style={"fontSize":"12px","color":COLORS["muted"],"marginTop":"3px"}) if subtitle else None
        ])
    ], style={"display":"flex","gap":"11px","alignItems":"center","marginBottom":"18px"})

def field(label, component):
    return html.Div([html.Label(label, style=LABEL), component], style={"marginBottom":"15px"})

def num_input(cid, value, mn, mx, step=1):
    return dcc.Input(id=cid, type="number", min=mn, max=mx, step=step, value=value, style=INPUT)

def dd(cid, options, value):
    return dcc.Dropdown(id=cid, options=options, value=value, clearable=False, searchable=False)

def metric_card(title, value_id, subtitle):
    return html.Div([
        html.Div(title.upper(), style={"fontSize":"10px","fontWeight":"800","letterSpacing":".08em","color":COLORS["muted"]}),
        html.Div(id=value_id, children="—", style={"fontSize":"23px","fontWeight":"800","color":COLORS["navy"],"marginTop":"6px"}),
        html.Div(subtitle, style={"fontSize":"11px","color":COLORS["muted"],"marginTop":"4px"})
    ], style={**CARD, "padding":"15px 17px", "flex":"1", "minWidth":"160px"})

def empty_figure(message="Les résultats apparaîtront après la prédiction."):
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=.5, y=.5,
                       showarrow=False, font=dict(size=14, color=COLORS["muted"]))
    fig.update_layout(template="plotly_white", paper_bgcolor="white",
                      plot_bgcolor="white", margin=dict(l=20,r=20,t=20,b=20), height=300)
    return fig

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div("SEPSIS", style={"fontSize":"12px","fontWeight":"900","letterSpacing":".16em","color":"#8EE3CF"}),
            html.Div("Early Warning AI", style={"fontSize":"25px","fontWeight":"800","color":"white","lineHeight":"1.1","marginTop":"3px"}),
            html.Div("Prototype de recherche en intelligence artificielle clinique",
                     style={"fontSize":"12px","color":"#C8D7E3","marginTop":"7px"})
        ]),
        html.Div("RESEARCH PROTOTYPE", style={
            "fontSize":"10px","fontWeight":"800","letterSpacing":".1em","color":"#D6F5ED",
            "border":"1px solid rgba(255,255,255,.25)","borderRadius":"999px",
            "padding":"8px 13px","backgroundColor":"rgba(255,255,255,.08)"
        })
    ], style={
        "background":f"linear-gradient(135deg,{COLORS['navy']} 0%,{COLORS['navy2']} 100%)",
        "padding":"22px 5%","display":"flex","justifyContent":"space-between",
        "alignItems":"center","boxShadow":"0 4px 20px rgba(18,48,74,.15)"
    }),

    html.Div([
        html.Div([
            html.Div([
                html.H1("Prédiction du risque de sepsis",
                        style={"fontSize":"29px","fontWeight":"800","color":COLORS["navy"],"margin":"0"}),
                html.Div("Estimation de l’apparition d’un sepsis dans les 6 prochaines heures",
                         style={"fontSize":"14px","color":COLORS["muted"],"marginTop":"7px"})
            ]),
            html.Div([
                html.Div("LANDMARK", style={"fontSize":"9px","fontWeight":"800","letterSpacing":".1em","color":COLORS["muted"]}),
                html.Div(id="active_landmark_badge", children="—",
                         style={"fontSize":"16px","fontWeight":"800","color":COLORS["teal"],"marginTop":"3px"})
            ], style={"backgroundColor":COLORS["teal_light"],"padding":"11px 17px","borderRadius":"12px",
                      "minWidth":"120px","textAlign":"center"})
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"center","gap":"20px","marginBottom":"18px"}),

        html.Div([
            section_header("01","Modèle de prédiction","Sélectionnez le modèle correspondant au point temporel souhaité."),
            dcc.Dropdown(id="model_selector", options=[{"label":k,"value":k} for k in models.keys()],
                         value=default_model, clearable=False),
            html.Div(id="model_window_hint", style={"fontSize":"12px","color":COLORS["teal"],"fontWeight":"600","marginTop":"9px"})
        ], style={**CARD,"padding":"22px 24px","marginBottom":"20px"}),

        html.Div([
            html.Div([
                section_header("02","Profil du patient","Informations démographiques."),
                field("Âge (années)", num_input("age",65,18,110)),
                field("Sexe", dd("gender",gender_options,"Male")),
                field("Groupe ethnique", dd("race",race_options,"White"))
            ], style={**CARD,"padding":"22px 24px"}),
            html.Div([
                section_header("03","Signes vitaux","Valeurs disponibles avant le landmark."),
                field("Fréquence cardiaque (bpm)", num_input("heart_rate",95,30,220)),
                field("SpO₂ (%)", num_input("spo2",96,50,100)),
                field("Température (°C)", num_input("temp",37.5,30,43,.1)),
                field("Fréquence respiratoire (/min)", num_input("rr",18,5,60)),
                field("Pression artérielle moyenne — MAP (mmHg)", num_input("map_ni",75,20,200))
            ], style={**CARD,"padding":"22px 24px"})
        ], style={"display":"grid","gridTemplateColumns":"minmax(0,.8fr) minmax(0,1.2fr)","gap":"20px","marginBottom":"20px"}),

        html.Div([
            section_header("04","Interventions et état clinique","Traitements et scores cliniques pertinents."),
            html.Div([
                field("Vasopresseur",dd("vaso",binary_options,0)),
                field("CRRT",dd("crrt",binary_options,0)),
                field("Ventilation invasive",dd("invasive",binary_options,0)),
                field("Ventilation non invasive",dd("noninv",binary_options,0)),
                field("Oxygène à haut débit",dd("highflow",binary_options,0)),
                field("Score de Glasgow",dd("gcs",gcs_options,3)),
                field("Comorbidité d’Elixhauser",dd("elix",elix_options,2))
            ], style={"display":"grid","gridTemplateColumns":"repeat(3,minmax(0,1fr))","gap":"0 18px"}),
            html.Div([
                html.Button("Analyser le risque de sepsis",id="predict_btn",n_clicks=0,style={
                    "border":"none","borderRadius":"10px",
                    "background":f"linear-gradient(135deg,{COLORS['teal']} 0%,#0B5D57 100%)",
                    "color":"white","fontSize":"14px","fontWeight":"800",
                    "padding":"13px 22px","cursor":"pointer",
                    "boxShadow":"0 5px 15px rgba(15,118,110,.22)"
                }),
                html.Div("Les valeurs doivent correspondre aux informations disponibles avant le landmark sélectionné.",
                         style={"fontSize":"11px","color":COLORS["muted"],"alignSelf":"center"})
            ], style={"display":"flex","gap":"16px","alignItems":"center","marginTop":"4px"})
        ], style={**CARD,"padding":"22px 24px","marginBottom":"20px"}),

        html.Div([
            section_header("05","Résultat de la prédiction","Estimation algorithmique du risque pour les 6 prochaines heures."),
            html.Div([
                metric_card("Probabilité de sepsis","metric_probability","Probabilité prédite"),
                metric_card("Catégorie de risque","metric_risk","Selon les seuils du prototype"),
                metric_card("Fenêtre prédite","metric_window","Après le landmark sélectionné")
            ], style={"display":"flex","gap":"14px","flexWrap":"wrap","marginBottom":"16px"}),
            html.Div(id="prediction_output", style={
                "padding":"12px 15px","borderRadius":"10px","backgroundColor":COLORS["blue_light"],
                "color":COLORS["navy"],"fontSize":"13px","fontWeight":"600","marginBottom":"12px"
            }),
            dcc.Graph(id="risk_gauge",figure=empty_figure(),config={"displayModeBar":False,"responsive":True})
        ], style={**CARD,"padding":"22px 24px","marginBottom":"20px"}),

        html.Div([
            html.Div([
                section_header("06","Trajectoire du risque","Évolution estimée sur les landmarks disponibles."),
                dcc.Graph(id="trajectory",figure=empty_figure(),config={"displayModeBar":False,"responsive":True})
            ], style={**CARD,"padding":"22px 24px","marginBottom":"20px"})
        ]),

        html.Div([
            html.Div("⚕",style={"fontSize":"22px","color":COLORS["amber"],"fontWeight":"800"}),
            html.Div([
                html.Div("AVERTISSEMENT — PROTOTYPE DE RECHERCHE",
                         style={"fontSize":"11px","fontWeight":"900","letterSpacing":".08em","color":COLORS["amber"]}),
                html.Div("Cet outil est destiné à la recherche et à des fins éducatives. "
                         "Il ne constitue pas un dispositif médical et ne remplace pas le jugement clinique.",
                         style={"fontSize":"12px","color":"#7C5A12","marginTop":"4px","lineHeight":"1.5"})
            ])
        ], style={"display":"flex","gap":"13px","alignItems":"flex-start","backgroundColor":"#FFFBEB",
                  "border":"1px solid #FDE68A","borderRadius":"13px","padding":"15px 18px","marginBottom":"22px"}),

        html.Div([
            html.Div("Sepsis Early Warning AI",style={"fontWeight":"800","color":COLORS["navy"]}),
            html.Div("Développé par Sangenis Ayao ASSOGBA • 2026",style={"marginTop":"4px"}),
            html.Div("Prototype de recherche — données et modèles à valider avant toute utilisation clinique.",style={"marginTop":"4px"})
        ], style={"textAlign":"center","fontSize":"11px","color":COLORS["muted"],"padding":"8px 0 35px"})
    ], style={"width":"90%","maxWidth":"1380px","margin":"0 auto","padding":"30px 0 0"})
], style=GLOBAL_STYLE)


# =========================================================
# PREDICTION CALLBACK
# =========================================================

@app.callback(
    [
        Output("prediction_output","children"),
        Output("risk_gauge","figure"),
        Output("trajectory","figure"),
        Output("metric_probability","children"),
        Output("metric_risk","children"),
        Output("metric_window","children"),
        Output("active_landmark_badge","children"),
        Output("model_window_hint","children")
    ],
    Input("predict_btn","n_clicks"),
    [
        State("model_selector","value"), State("age","value"), State("gender","value"),
        State("race","value"), State("heart_rate","value"), State("spo2","value"),
        State("temp","value"), State("rr","value"), State("map_ni","value"),
        State("vaso","value"), State("crrt","value"), State("invasive","value"),
        State("noninv","value"), State("highflow","value"), State("gcs","value"),
        State("elix","value")
    ]
)
def predict(n_clicks,model_name,age,gender,race,hr,spo2,temp,rr,map_ni,
            vaso,crrt,invasive,noninv,highflow,gcs,elix):

    if n_clicks is None or not model_name or model_name not in models:
        return ("Saisissez les paramètres du patient puis lancez l’analyse.",
                empty_figure(),
                empty_figure("La trajectoire apparaîtra après la prédiction."),
                "—","—","—","—","")

    errors = validate_inputs(age,hr,spo2,temp,rr,map_ni)
    if errors:
        return (" | ".join(errors),empty_figure("Vérifiez les valeurs saisies."),
                empty_figure("Analyse indisponible."),
                "—","Erreur","—",WINDOW_LABELS.get(model_name,"—"),
                "Veuillez corriger les valeurs signalées avant de relancer l’analyse.")

    X = pd.DataFrame([{
        "anchor_age":age,"heart_rate_max":hr,"spo2_min":spo2,
        "temperature_max":temp,"rr_max":rr,"map_ni":map_ni,
        "vasopressor_mean":vaso,"crrt_mean":crrt,"invasive_mean":invasive,
        "noninvasive_mean":noninv,"highflow_mean":highflow,
        "gender":gender,"race_grouped":race,
        "gcs_min_cat_num":gcs,"elixhauser_cat_num":elix,
        "gcs_min_missing":0,"map_ni_missing":0
    }])

    model = models[model_name]
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception as err:
        return (f"Erreur de prédiction : {str(err)}",empty_figure("Erreur de calcul."),
                empty_figure("Erreur de calcul."),empty_figure("Erreur de calcul."),
                "—","Erreur","—",WINDOW_LABELS.get(model_name,"—"),
                "Le modèle n’a pas pu traiter les paramètres fournis.")

    risk = "LOW" if prob < .30 else "MODERATE" if prob < .60 else "HIGH"
    risk_fr = {"LOW":"FAIBLE","MODERATE":"MODÉRÉ","HIGH":"ÉLEVÉ"}[risk]
    window = WINDOW_LABELS.get(model_name,"—")
    landmark = model_name.split(" ")[0].replace("(","").replace(")","")
    text = f"Probabilité estimée : {prob:.1%}  •  Risque : {risk_fr}  •  Fenêtre : {window}"

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",value=prob,
        number={"valueformat":".1%","font":{"size":42,"color":COLORS["navy"]}},
        gauge={
            "axis":{"range":[0,1],"tickformat":".0%","tickfont":{"color":COLORS["muted"]}},
            "bar":{"color":COLORS["teal"],"thickness":.25},"bgcolor":"#F8FAFC",
            "borderwidth":0,
            "steps":[{"range":[0,.30],"color":"#DCFCE7"},
                     {"range":[.30,.60],"color":"#FEF3C7"},
                     {"range":[.60,1],"color":"#FEE2E2"}],
            "threshold":{"line":{"color":COLORS["navy"],"width":3},"thickness":.75,"value":prob}
        },
        title={"text":"Probabilité prédite de sepsis","font":{"size":15,"color":COLORS["navy"]}}
    ))
    gauge.update_layout(height=330,margin=dict(l=35,r=35,t=55,b=20),
                        paper_bgcolor="white",font=dict(family="Inter, Segoe UI, Arial"))
    start_index = LANDMARK_ORDER.index(model_name) if model_name in LANDMARK_ORDER else 0
    probs,windows = [],[]
    for lm in LANDMARK_ORDER[start_index:]:
        if lm in models:
            try:
                probs.append(float(models[lm].predict_proba(X)[0][1]))
                windows.append(WINDOW_LABELS[lm])
            except Exception:
                pass

    traj = go.Figure()
    if probs:
        traj.add_trace(go.Scatter(
            x=windows,y=probs,mode="lines+markers",
            line=dict(width=3,color=COLORS["teal"]),
            marker=dict(size=10,color=COLORS["navy"],line=dict(width=2,color="white")),
            hovertemplate="<b>%{x}</b><br>Probabilité : %{y:.1%}<extra></extra>"
        ))
    traj.update_layout(
        title={"text":"Trajectoire estimée du risque","font":{"size":16,"color":COLORS["navy"]}},
        yaxis={"range":[0,1],"tickformat":".0%","title":"Probabilité de sepsis",
               "gridcolor":"#E8EEF3","zeroline":False},
        xaxis={"title":"Fenêtre de prédiction","gridcolor":"#F1F5F9"},
        template="plotly_white",paper_bgcolor="white",plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial"),
        margin=dict(l=65,r=25,t=55,b=55),hovermode="x unified"
    )

    return (text,gauge,traj,f"{prob:.1%}",risk_fr,window,landmark,
            f"Landmark sélectionné : {landmark} → prédiction pour {window}.")


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, jupyter_mode="external")
