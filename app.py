import math
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from arch import arch_model
from scipy.stats import norm as scinorm
import matplotlib.pyplot as plt
import shap

# ---------------------------
#  CONFIG
# ---------------------------
config = {
    "SEQ_LENGTH":         20,
    "HORIZON":             1,
    "DEVICE":  torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "d_model":            64,
    "nhead":               2,
    "num_encoder_layers":  2,
    "kafin_hidden":       32
}

# ---------------------------
#  PROVIDED MODEL DEFINITIONS
#  (CNN-Attention-MLP, InformerModel, KAFINPricer, OptionPricingPipeline,
#   load_models(), load_stats())
# ---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class CNN_Attention_MLP(nn.Module):
    def __init__(self, window, horizon, feat_dim=6):
        super().__init__()
        self.conv1 = nn.Conv1d(feat_dim, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.posenc = PositionalEncoding(d_model=256, max_len=window)
        self.attn   = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(256 * window, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),           nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128),            nn.ReLU(),
            nn.Linear(128, horizon)
        )

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = F.relu(self.bn2(self.conv2(o)))
        o = F.relu(self.bn3(self.conv3(o)))      # (B,256,window)
        attn_in = o.transpose(1, 2)               # (B,window,256)
        attn_in = self.posenc(attn_in)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        o2 = attn_out + attn_in                   # (B,window,256)
        flat = o2.reshape(o2.size(0), -1)         # (B,256*window)
        return self.fc(flat)

class OptionPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)
    def forward(self, specs):
        strike, underlying, ttm = specs[...,0], specs[...,1], specs[...,2]
        log_m = torch.log((underlying + 1e-6)/(strike + 1e-6))
        feats = torch.stack([log_m, ttm], dim=-1)
        return self.linear(feats)

class InformerModel(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, num_encoder_layers):
        super().__init__()
        self.input_linear = nn.Linear(feature_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.option_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.output_linear = nn.Linear(d_model,1)
        self.attn_norm = nn.LayerNorm(d_model)
    def forward(self, x, time_pe, option_pe, option_key_padding_mask=None):
        B,T,N,F = x.shape
        dev = x.device
        x = self.input_linear(x) \
          + time_pe.to(dev).unsqueeze(0).unsqueeze(2) \
          + option_pe.to(dev).unsqueeze(1)
        x = x.permute(0,2,1,3).contiguous().view(B*N,T,-1).transpose(0,1)
        x = self.transformer_encoder(x).transpose(0,1)[:,-1,:].view(B,N,-1)
        if option_key_padding_mask is not None:
            option_key_padding_mask = option_key_padding_mask.bool()
        x_fp32 = x.float()
        with torch.cuda.amp.autocast(enabled=False):
            attn_out,_ = self.option_attn(x_fp32, x_fp32, x_fp32, key_padding_mask=option_key_padding_mask)
        attn_out = attn_out.to(x.dtype)
        x = x + attn_out
        x = self.attn_norm(x)
        return self.output_linear(x)

class KAFINBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.h_funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])
        self.attn_query = nn.Linear(input_dim, hidden_dim)
        self.attn_key   = nn.Linear(input_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, input_dim)
        self.cross      = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))
        self.g_func     = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))
    def forward(self,x):
        h_outs = [fn(x[:,j:j+1]) for j,fn in enumerate(self.h_funcs)]
        h_cat  = torch.cat(h_outs, dim=1)
        q,k    = self.attn_query(h_cat), self.attn_key(h_cat)
        scores = self.attn_score(torch.tanh(q*k))
        wts    = torch.softmax(scores, dim=1)
        weighted = (h_cat * wts).sum(1,keepdim=True)
        cross = self.cross(h_cat)
        comb  = weighted + cross
        return self.g_func(comb)

class KAFINPricer(nn.Module):
    def __init__(self,input_dim=5,hidden_dim=32,L2=3):
        super().__init__()
        self.blocks = nn.ModuleList([KAFINBlock(input_dim, hidden_dim) for _ in range(L2)])
    def forward(self,x):
        B,N,D = x.shape
        flat = x.view(B*N,D)
        outs = [blk(flat) for blk in self.blocks]
        total = torch.stack(outs,dim=2).sum(2)
        return total.view(B,N,1)

class OptionPricingPipeline(nn.Module):
    def __init__(self, informer, kafin):
        super().__init__()
        self.inf = informer
        self.kaf = kafin
        self.option_pe = OptionPositionalEncoding(config["d_model"]).to(config["DEVICE"])
    def forward(self, seq, tp, specs, option_mask):
        op = self.option_pe(specs)
        iv = self.inf(seq, tp, op, option_key_padding_mask=option_mask)
        price = self.kaf(torch.cat([specs, iv], dim=-1))
        return iv, price

@st.cache_resource
def load_models():
    mlp = CNN_Attention_MLP(window=config["SEQ_LENGTH"], horizon=config["HORIZON"], feat_dim=6)
    mlp.load_state_dict(torch.load('models/bestMLPmodel.pth', map_location='cpu'))
    mlp.to(config["DEVICE"]).eval()
    inf = InformerModel(feature_dim=14, d_model=config["d_model"],
                        nhead=config["nhead"], num_encoder_layers=config["num_encoder_layers"])
    kaf = KAFINPricer(input_dim=5, hidden_dim=config["kafin_hidden"], L2=3)
    pipeline = OptionPricingPipeline(inf, kaf)
    ckpt = torch.load('models/bestITMATMmodel.pth', map_location='cpu')
    pipeline.load_state_dict(ckpt)
    pipeline.to(config["DEVICE"]).eval()
    return mlp, pipeline

@st.cache_data
def load_stats():
    df = pd.read_csv('data/summary_stats.csv')
    stats = {}
    fmap = {'underlying_price':'UNDERLYING_LAST','strike':'STRIKE',
            'time_to_maturity':'time_to_maturity','IV':'IV'}
    for _,row in df.iterrows():
        if row['feature'] in fmap:
            stats[fmap[row['feature']]] = (row['mean'], row['std'])
    return stats

# ---------------------------
#  STREAMLIT APP
# ---------------------------

st.title("One-Step-Ahead Option Pricing & XAI: TSLA")
mlp_model, pipeline_model = load_models()
stats = load_stats()

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker", value="TSLA")
K      = st.sidebar.number_input("Strike Price (K)", value=250.0)
T      = st.sidebar.number_input("Time to Maturity (years)", value=0.25)
r      = st.sidebar.number_input("Risk-Free Rate (r)", value=0.015)
IV_in  = st.sidebar.number_input("Implied Volatility (for MLP/Informer)", value=0.55)
model_choice = st.sidebar.selectbox("Model", ["MLP","Informer+KAFIN","GARCH+Black-Scholes"])

# Fetch last 20 trading days
data = yf.Ticker(ticker).history(period="1mo").dropna().iloc[-config["SEQ_LENGTH"]:]
last_date = data.index[-1].date()
pred_date = last_date + pd.Timedelta(days=1)
st.sidebar.markdown(f"Last data date: **{last_date}**  \nPredicting: **{pred_date}**")

# Derived from last day
S_last = float(data["Close"].iloc[-1])
moneyness = S_last / K
logm      = math.log(moneyness + 1e-8)

st.subheader(f"Input parameters from data:")
st.write({
    "Underlying (S)":    S_last,
    "Strike (K)":        K,
    "Time to Maturity":  T,
    "Interest Rate (r)": r,
    "Implied Volatility": IV_in if model_choice!="GARCH+Black-Scholes" else "From GARCH",
    "Moneyness":         moneyness,
    "Log-Moneyness":     logm
})

def norm(col, x):
    m,s = stats[col]
    return (x - m)/s

def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S-K,0.0)
    d1 = (math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return float(S*scinorm.cdf(d1) - K*math.exp(-r*T)*scinorm.cdf(d2))

# Build MLP input sequence (6×20)
inp = np.zeros((6, config["SEQ_LENGTH"]), dtype=np.float32)
for i in range(config["SEQ_LENGTH"]):
    u = data["Close"].iloc[i]
    up = norm("UNDERLYING_LAST", u)
    stn= norm("STRIKE",          K)
    ttn= norm("time_to_maturity",T)
    ivn= norm("IV",              IV_in)
    mon= u/K
    logm_i = math.log(mon+1e-8)
    inp[:, i] = [up, stn, ttn, mon, logm_i, ivn]
mlp_seq = torch.tensor(inp[None], dtype=torch.float32, device=config["DEVICE"])

# Build pipeline sequence (14 features)
feat14 = np.zeros((config["SEQ_LENGTH"], 14), dtype=np.float32)
for i in range(config["SEQ_LENGTH"]):
    u = data["Close"].iloc[i]
    up = norm("UNDERLYING_LAST", u)
    stn= norm("STRIKE", K)
    ttn= norm("time_to_maturity", T)
    ivn= norm("IV", IV_in)
    mon= u/K
    logm_i = math.log(mon+1e-8)
    feat14[i,:6] = [up, stn, ttn, mon, logm_i, ivn]
seq_full = torch.tensor(feat14.reshape(1,config["SEQ_LENGTH"],1,14),
                       dtype=torch.float32, device=config["DEVICE"])
# Time positional encoding
time_pe = torch.zeros(config["SEQ_LENGTH"], config["d_model"], device=config["DEVICE"])
pos   = torch.arange(0,config["SEQ_LENGTH"],dtype=torch.float32,device=config["DEVICE"]).unsqueeze(1)
div   = torch.exp(torch.arange(0,config["d_model"],2,device=config["DEVICE"])*(-math.log(10000.0)/config["d_model"]))
time_pe[:,0::2] = torch.sin(pos*div)
time_pe[:,1::2] = torch.cos(pos*div)
# Specs & mask
specs = torch.tensor([[[K, S_last, T, r, IV_in]]], dtype=torch.float32, device=config["DEVICE"])
option_mask = torch.zeros(1,1, dtype=torch.bool, device=config["DEVICE"])

# Feature list for PDP
feat_names = ["Underlying Price","Strike Price","Time to Maturity","Moneyness","Log‑Moneyness","Implied Volatility"]
pdp_feature = st.sidebar.selectbox("PDP feature", feat_names)

# Helper for PDP grid
def make_grid(feature, base):
    return np.linspace(base*0.5, base*1.5, 30)

# Prediction & XAI
if model_choice == "MLP":
    with torch.no_grad():
        logp = mlp_model(mlp_seq)[0,0].item()
    price = math.exp(logp)
    st.subheader(f"MLP one-step price: {price:.4f}")

    # SHAP via gradient×input on last timestep
    mlp_seq.requires_grad_()
    mlp_model.zero_grad()
    out = mlp_model(mlp_seq)
    out.backward()
    grads = mlp_seq.grad[0,:, -1].cpu().numpy()
    inputs_last = inp[:, -1]
    shap_vals = np.abs(inputs_last * grads)
    fig, ax = plt.subplots(figsize=(7,3))
    ax.bar(feat_names, shap_vals)
    ax.set_xticklabels(feat_names, rotation=45, ha="right")
    ax.set_ylabel("|grad×input|")
    ax.set_title("MLP Feature Importances (last day)")
    plt.tight_layout()
    st.pyplot(fig)

    # PDP
    base_vals = {
        "Underlying Price":  S_last,
        "Strike Price":      K,
        "Time to Maturity":  T,
        "Moneyness":         moneyness,
        "Log‑Moneyness":     logm,
        "Implied Volatility": IV_in
    }
    grid = make_grid(pdp_feature, base_vals[pdp_feature])
    pdp = []
    for v in grid:
        inp2 = inp.copy()
        if pdp_feature == "Underlying Price":
            inp2[0,-1] = norm("UNDERLYING_LAST", v)
            inp2[3,-1] = v/K
            inp2[4,-1] = math.log(v/K+1e-8)
        elif pdp_feature == "Strike Price":
            inp2[1,-1] = norm("STRIKE", v)
            inp2[3,-1] = S_last/v
            inp2[4,-1] = math.log(S_last/v+1e-8)
        elif pdp_feature == "Time to Maturity":
            inp2[2,-1] = norm("time_to_maturity", v)
        elif pdp_feature == "Moneyness":
            inp2[3,-1] = v
            inp2[4,-1] = math.log(v+1e-8)
            inp2[0,-1] = norm("UNDERLYING_LAST", v*K)
        elif pdp_feature == "Log‑Moneyness":
            inp2[4,-1] = v
            s_mod = math.exp(v)*K
            inp2[0,-1] = norm("UNDERLYING_LAST", s_mod)
            inp2[3,-1] = s_mod/K
        else:  # IV
            inp2[5,-1] = norm("IV", v)
        seq2 = torch.tensor(inp2[None], dtype=torch.float32, device=config["DEVICE"])
        with torch.no_grad():
            lp2 = mlp_model(seq2)[0,0].item()
        pdp.append(math.exp(lp2))
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(grid, pdp, linewidth=2)
    ax2.set_xlabel(pdp_feature)
    ax2.set_ylabel("Price")
    ax2.set_title(f"PDP: {pdp_feature}")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

elif model_choice == "Informer+KAFIN":
    # --- Base prediction and header ---
    with torch.no_grad():
        iv_pred, price_pred = pipeline_model(seq_full, time_pe, specs, option_mask)
    ivv = iv_pred.item()
    pr = price_pred.item()
    st.subheader(f"Informer IV: {ivv:.4f}")
    st.subheader(f"Informer+KAFIN Price: {pr:.4f}")
    st.info("SHAP not available for this model.")

    # --- Prepare for PDP ---
    base_vals = {
        "Underlying Price":   S_last,
        "Strike Price":       K,
        "Time to Maturity":   T,
        "Interest Rate":      r,
        "Implied Volatility": IV_in,
        "Moneyness":          moneyness,
        "Log‑Moneyness":      logm
    }
    base_feat14 = feat14.copy()   # shape (20,14)
    base_specs  = specs.clone()   # shape (1,1,5)

    grid = np.linspace(
        base_vals[pdp_feature] * 0.5,
        base_vals[pdp_feature] * 1.5,
        30
    )
    pdp_prices = []

    for v in grid:
        f2    = base_feat14.copy()
        spec2 = base_specs.clone()

        if pdp_feature == "Underlying Price":
            # sequence columns
            f2[-1, 0] = norm("UNDERLYING_LAST", v)
            f2[-1, 3] = v / K
            f2[-1, 4] = np.log(v / K + 1e-8)
            # specs: update underlying
            spec2[0,0,1] = v

        elif pdp_feature == "Strike Price":
            f2[-1, 1] = norm("STRIKE", v)
            f2[-1, 3] = S_last / v
            f2[-1, 4] = np.log(S_last / v + 1e-8)
            spec2[0,0,0] = v

        elif pdp_feature == "Time to Maturity":
            f2[-1, 2] = norm("time_to_maturity", v)
            spec2[0,0,2] = v

        elif pdp_feature == "Interest Rate":
            spec2[0,0,3] = v

        elif pdp_feature == "Implied Volatility":
            f2[-1, 5] = norm("IV", v)
            spec2[0,0,4] = v

        elif pdp_feature == "Moneyness":
            # new mon, log-mon in sequence
            f2[-1, 3] = v
            f2[-1, 4] = np.log(v + 1e-8)
            # derive synthetic underlying for sequence & specs
            S_mod = v * K
            f2[-1, 0] = norm("UNDERLYING_LAST", S_mod)
            spec2[0,0,1] = S_mod

        elif pdp_feature == "Log‑Moneyness":
            f2[-1, 4] = v
            mon_new = np.exp(v)
            f2[-1, 3] = mon_new
            S_mod = mon_new * K
            f2[-1, 0] = norm("UNDERLYING_LAST", S_mod)
            spec2[0,0,1] = S_mod

        # --- Run pipeline with modified inputs ---
        seq2 = torch.tensor(
            f2.reshape(1, config["SEQ_LENGTH"], 1, 14),
            dtype=torch.float32,
            device=config["DEVICE"]
        )
        with torch.no_grad():
            _, price2 = pipeline_model(seq2, time_pe, spec2, option_mask)
        pdp_prices.append(price2.item())

    # --- Plot PDP (after loop) ---
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(grid, pdp_prices, linewidth=2)
    ax.set_xlabel(pdp_feature)
    ax.set_ylabel("Predicted Option Price")
    ax.set_title(f"PDP for Informer+KAFIN: {pdp_feature}")
    ax.grid(alpha=0.3)
    st.pyplot(fig)


else:  # GARCH + Black-Scholes
    returns = data["Close"].pct_change().dropna()
    am = arch_model(returns, vol="Garch", p=1, q=1)
    res = am.fit(disp="off")
    fcast = res.forecast(horizon=1).variance.iloc[-1,0]
    sigma_d = math.sqrt(fcast)
    sigma_a = sigma_d * math.sqrt(252)
    price_g = bs_call_price(S_last, K, T, r, sigma_a)
    st.subheader(f"GARCH+BS Price: {price_g:.4f}")
    st.info("SHAP not available for this model.")

    # PDP for GARCH+BS
    base_vals = {
        "Underlying Price":  S_last,
        "Strike Price":      K,
        "Time to Maturity":  T,
        "Moneyness":         moneyness,
        "Log‑Moneyness":     logm,
        "Implied Volatility": sigma_a
    }
    grid = make_grid(pdp_feature, base_vals[pdp_feature])
    pdp = []
    for v in grid:
        if pdp_feature == "Underlying Price":
            pdp.append(bs_call_price(v, K, T, r, sigma_a))
        elif pdp_feature == "Strike Price":
            pdp.append(bs_call_price(S_last, v, T, r, sigma_a))
        elif pdp_feature == "Time to Maturity":
            pdp.append(bs_call_price(S_last, K, v, r, sigma_a))
        elif pdp_feature == "Moneyness":
            pdp.append(bs_call_price(v*K, K, T, r, sigma_a))
        elif pdp_feature == "Log‑Moneyness":
            pdp.append(bs_call_price(math.exp(v)*K, K, T, r, sigma_a))
        else:  # IV
            pdp.append(bs_call_price(S_last, K, T, r, v))
    fig4, ax4 = plt.subplots(figsize=(8,5))
    ax4.plot(grid, pdp, linewidth=2)
    ax4.set_xlabel(pdp_feature)
    ax4.set_ylabel("Price")
    ax4.set_title(f"PDP: {pdp_feature}")
    ax4.grid(alpha=0.3)
    st.pyplot(fig4)
