import math
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt

from scipy.stats import norm

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
#  MODEL DEFINITIONS
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
        # x: (B, L, D)
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
        # x: (B, feat_dim, window)
        o = F.relu(self.bn1(self.conv1(x)))
        o = F.relu(self.bn2(self.conv2(o)))
        o = F.relu(self.bn3(self.conv3(o)))      # (B,256,window)

        attn_in = o.transpose(1, 2)              # (B,window,256)
        attn_in = self.posenc(attn_in)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        o2 = attn_out + attn_in                  # (B,window,256)

        flat = o2.reshape(o2.size(0), -1)        # (B,256*window)
        return self.fc(flat)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, D)
        return self.pe[: x.size(1), :]

class OptionPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)

    def forward(self, specs):
        # specs: (B, N, 5) → strike, underlying, ttm, r, IV
        strike, underlying, ttm = specs[..., 0], specs[..., 1], specs[..., 2]
        log_m = torch.log((underlying + 1e-6) / (strike + 1e-6))
        feats = torch.stack([log_m, ttm], dim=-1)  # (B,N,2)
        return self.linear(feats)                 # (B,N,d_model)

class InformerModel(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, num_encoder_layers):
        super().__init__()
        self.input_linear = nn.Linear(feature_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.option_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.output_linear = nn.Linear(d_model, 1)
        self.attn_norm = nn.LayerNorm(d_model)

    def forward(self, x, time_pe, option_pe, option_key_padding_mask=None):
        B, T, N, F = x.shape
        dev = x.device

        # add positional encodings
        x = self.input_linear(x) \
            + time_pe.to(dev).unsqueeze(0).unsqueeze(2) \
            + option_pe.to(dev).unsqueeze(1)

        # (T, B*N, d_model)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(0, 1)[:, -1, :].view(B, N, -1)

        if option_key_padding_mask is not None:
            option_key_padding_mask = option_key_padding_mask.bool()

        # full-precision attention
        x_fp32 = x.float()
        with torch.cuda.amp.autocast(enabled=False):
            attn_out, _ = self.option_attn(x_fp32, x_fp32, x_fp32, key_padding_mask=option_key_padding_mask)
        attn_out = attn_out.to(x.dtype)

        x = x + attn_out
        x = self.attn_norm(x)
        return self.output_linear(x)  # (B,N,1)

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
        self.attn_query    = nn.Linear(input_dim, hidden_dim)
        self.attn_key    = nn.Linear(input_dim, hidden_dim)
        self.attn_score= nn.Linear(hidden_dim, input_dim)
        self.cross     = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))
        self.g_func    = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))

    def forward(self, x):
        # x: (B*N, D)
        h_outs = [fn(x[:, j:j+1]) for j, fn in enumerate(self.h_funcs)]
        h_cat  = torch.cat(h_outs, dim=1)             # (B*N, D)
        q, k   = self.attn_query(h_cat), self.attn_key(h_cat)
        scores = self.attn_score(torch.tanh(q * k))
        wts    = torch.softmax(scores, dim=1)
        weighted= (h_cat * wts).sum(dim=1, keepdim=True)
        cross  = self.cross(h_cat)
        comb   = weighted + cross
        return self.g_func(comb)                      # (B*N,1)

class KAFINPricer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, L2=3):
        super().__init__()
        self.blocks = nn.ModuleList([KAFINBlock(input_dim, hidden_dim) for _ in range(L2)])

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(B * N, D)
        outs   = [blk(x_flat) for blk in self.blocks]
        total  = torch.stack(outs, dim=2).sum(2)       # (B*N,1)
        return total.view(B, N, 1)                    # (B,N,1)

class OptionPricingPipeline(nn.Module):
    def __init__(self, informer, kafin):
        super().__init__()
        self.inf   = informer
        self.kaf   = kafin
        self.option_pe = OptionPositionalEncoding(config["d_model"]).to(config["DEVICE"])

    def forward(self, seq, tp, specs, option_mask):
        op = self.option_pe(specs)
        iv = self.inf(seq, tp, op, option_key_padding_mask=option_mask)
        price = self.kaf(torch.cat([specs, iv], dim=-1))
        return iv, price

# ---------------------------
#  LOAD MODELS & STATS
# ---------------------------

@st.cache_resource
def load_models():
    # MLP
    mlp = CNN_Attention_MLP(window=config["SEQ_LENGTH"], horizon=config["HORIZON"], feat_dim=6)
    mlp.load_state_dict(torch.load(
        r'C:\Users\ryan9\Documents\Projects\XAI\XAI_Project_App\models\bestMLPmodel.pth',
        map_location='cpu'
    ))
    mlp = mlp.to(config["DEVICE"]).eval()

    # Informer+KAFIN
    inf  = InformerModel(
        feature_dim=14,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"]
    )
    kaf  = KAFINPricer(input_dim=5, hidden_dim=config["kafin_hidden"], L2=3)
    pipeline = OptionPricingPipeline(inf, kaf)

    ckpt = torch.load(
        r'C:\Users\ryan9\Documents\Projects\XAI\XAI_Project_App\models\bestITMATMmodel.pth',
        map_location='cpu'
    )
    pipeline.load_state_dict(ckpt)
    pipeline = pipeline.to(config["DEVICE"]).eval()

    return mlp, pipeline

@st.cache_data
def load_stats():
    # Load summary statistics from CSV file
    stats_df = pd.read_csv(
        r"C:\Users\ryan9\Documents\Projects\XAI\XAI_Project_App\data\summary_stats.csv"
    )
    
    # Convert to dictionary mapping
    stats = {}
    # Map feature names to what the app expects
    feature_map = {
        'underlying_price': 'UNDERLYING_LAST',
        'strike': 'STRIKE',
        'time_to_maturity': 'time_to_maturity',
        'IV': 'IV'
    }
    
    for _, row in stats_df.iterrows():
        feature = row['feature']
        if feature in feature_map:
            key = feature_map[feature]
            stats[key] = (row['mean'], row['std'])
    
    return stats

mlp_model, pipeline_model = load_models()
stats = load_stats()

# ---------------------------
#  STREAMLIT APP
# ---------------------------

st.title("Option Pricing & Explainability Demo")
st.sidebar.header("Option Parameters")

S  = st.sidebar.number_input("Underlying Price (S)",     value=260.0)
K  = st.sidebar.number_input("Strike Price (K)",         value=250.0)
T  = st.sidebar.number_input("Time to Maturity (years)", value=0.25)
IV = st.sidebar.number_input("Implied Volatility",       value=0.55)
r  = st.sidebar.number_input("Interest Rate",            value=0.015)

model_choice = st.sidebar.selectbox("Model", ["MLP","Informer+KAFIN"])

def norm(col,x):
    m,s = stats[col]
    return (x-m)/s

# --- MLP inputs ---
up      = norm("UNDERLYING_LAST", S)
st_n    = norm("STRIKE",          K)
tt_n    = norm("time_to_maturity",T)
iv_n    = norm("IV",              IV)
mon     = S/K
logm    = np.log(mon+1e-8)
mlp_inputs = np.array([up,st_n,tt_n,mon,logm,iv_n],dtype=np.float32)

# pad to 20 timesteps
inp = np.zeros((6,config["SEQ_LENGTH"]),dtype=np.float32)
inp[:,0] = mlp_inputs
mlp_tensor = torch.tensor(inp[None,:,:], dtype=torch.float32, device=config["DEVICE"])  # (1,6,20)

# --- Pipeline inputs ---
# Combine MLP inputs with additional features - total should be 14 to match model's feature_dim=14
feat14 = np.concatenate([mlp_inputs, np.zeros(8, dtype=np.float32)])
# Shape needs to be (B=1, T=1, N=1, F=14) to match model initialization
seq_full = torch.tensor(feat14.reshape(1, 1, 1, 14), 
                       dtype=torch.float32,
                       device=config["DEVICE"])
# Option specs with dimensions B=1, N=1, specs=5 (strike, underlying, ttm, r, IV)
specs = torch.tensor([[[K, S, T, r, IV]]], 
                     dtype=torch.float32,
                     device=config["DEVICE"])

if model_choice=="MLP":
    with torch.no_grad():
        y = mlp_model(mlp_tensor).item()
    price = np.exp(y)
    st.subheader(f"MLP Price: {price:.4f}")

    # --- SHAP Force Plot (Single Sample Only) ---
    background = mlp_tensor.repeat(50, 1, 1)
    explainer = shap.DeepExplainer(mlp_model, background)
    sv = explainer.shap_values(mlp_tensor)

    # Extract first sample's values and flatten to (N,) shape
    shap_vals = sv[0][0].flatten()
    base_val = explainer.expected_value[0]

    st.subheader("SHAP Force Plot (Single Sample)")
    fig1, ax1 = plt.subplots()
    shap.plots.force(
        base_value=base_val,
        shap_values=shap_vals,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig1)

    # PDP/ICE
    feat_names = ["Underlying Price","Strike Price","Time to Maturity","Moneyness","Log-Moneyness","Implied Volatility"]
    sel = st.sidebar.selectbox("Feature for PDP/ICE", feat_names)
    idx = feat_names.index(sel)
    grid = np.linspace(mlp_inputs[idx]*0.5, mlp_inputs[idx]*1.5, 50)
    pdp_vals, ice_vals = [], []
    
    for val in grid:
        arr = mlp_inputs.copy()
        arr[idx] = val
        # Reshape properly to (1,6,20)
        padded = np.zeros((6, config["SEQ_LENGTH"]), dtype=np.float32)
        padded[:, 0] = arr
        t = torch.tensor(padded[None, :, :], dtype=torch.float32, device=config["DEVICE"])
        
        with torch.no_grad():
            out = mlp_model(t).item()
        pdp_vals.append(np.exp(out))
        ice_vals.append(np.exp(out))
        
    fig2, ax2 = plt.subplots()
    ax2.plot(grid, pdp_vals, label="PDP")
    ax2.plot(grid, ice_vals, "--", label="ICE")
    ax2.legend()
    ax2.set_title(f"PDP & ICE for {sel}")
    st.pyplot(fig2)

else:
    try:
        # Create time positional encoding exactly as in notebook
        T = seq_full.shape[1]  # Should be 1
        
        # Print shapes for debugging
        st.write(f"seq_full shape: {seq_full.shape}, specs shape: {specs.shape}")
        
        # Create positional encoding from scratch to match notebook exactly
        d_model = config["d_model"]
        pe = torch.zeros(T, d_model)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        st.write(f"pe shape: {pe.shape}")
        
        # Move pe to the right device
        pe = pe.to(config["DEVICE"])
        
        # Create an option mask where False means no padding (all valid)
        option_mask = torch.zeros(1, 1, dtype=torch.bool, device=config["DEVICE"])
        
        with torch.no_grad():
            try:
                iv_pred, pr_pred = pipeline_model(seq_full, pe, specs, option_mask=option_mask)
                st.subheader(f"Informer IV: {iv_pred.item():.4f}")
                st.subheader(f"Informer+KAFIN Price: {pr_pred.item():.4f}")
            except Exception as e1:
                st.error(f"First attempt failed: {str(e1)}")
                # Try with unsqueezed pe
                try:
                    iv_pred, pr_pred = pipeline_model(seq_full, pe.unsqueeze(0), specs, option_mask=option_mask)
                    st.subheader(f"Informer IV: {iv_pred.item():.4f}")
                    st.subheader(f"Informer+KAFIN Price: {pr_pred.item():.4f}")
                except Exception as e2:
                    st.error(f"Second attempt failed: {str(e2)}")
                    raise e2

        # show block contributions
        kaf_in = torch.cat([specs, iv_pred], dim=-1).view(1, -1)
        with torch.no_grad():  # Add this to prevent gradient tracking
            blocks = [blk(kaf_in) for blk in pipeline_model.kaf.blocks]
            contribs = torch.stack(blocks, dim=2).sum(2).detach().cpu().numpy().flatten()
        
        fig3, ax3 = plt.subplots()
        ax3.bar(range(len(contribs)), contribs)
        ax3.set_title("KAFIN Block Contributions")
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Error in Informer+KAFIN model: {type(e).__name__}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
