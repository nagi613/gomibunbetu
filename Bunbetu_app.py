import streamlit as st
from PIL import Image
import torch
pip install torch

# 分別カテゴリ（ラベル → 日本語＋アイコン）
label_to_category = {
    "bottle": "資源ゴミ（ペットボトル）🧴",
    "can": "資源ゴミ（缶）🥫",
    "cup": "燃えるゴミ（紙コップ）☕",
    "banana": "燃えるゴミ（生ゴミ）🍌",
    "apple": "燃えるゴミ（生ゴミ）🍎",
    "orange": "燃えるゴミ（生ゴミ）🍊",
    "broccoli": "燃えるゴミ（野菜）🥦",
    "carrot": "燃えるゴミ（野菜）🥕",
    "pizza": "燃えるゴミ（食べ残し）🍕",
    "cake": "燃えるゴミ（食べ残し）🍰",
    "laptop": "燃えないゴミ（電子機器）💻",
    "cell phone": "燃えないゴミ（バッテリー）📱",
    "tv": "燃えないゴミ（テレビ）📺",
    "microwave": "燃えないゴミ（電子レンジ）🔥",
    "vase": "燃えないゴミ（陶器）🏺",
    "book": "資源ゴミ（紙類）📚",
    "newspaper": "資源ゴミ（新聞紙）🗞️",
    "toilet paper": "資源ゴミ（紙類）🧻",
    "dining table": "粗大ごみ（家具）🛋️",
    "chair": "粗大ごみ（椅子）🪑",
    "bed": "粗大ごみ（ベッド）🛏️",
    "person": "分別不可（人間）🚫",
    "handbag": "燃えないゴミ（革・布）👜",
    "shoe": "燃えるゴミ（靴）👞",
    "backpack": "燃えるゴミ（布類）🎒",
    "clock": "燃えないゴミ（機械）⏰",
    "umbrella": "燃えないゴミ（傘）🌂",
    "scissors": "燃えないゴミ（金属）✂️",
    "toothbrush": "燃えないゴミ（プラスチック）🪥"
}


# YOLOv5 モデルの読み込み
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5n', source='github')

model = load_model()

# タイトルと説明
st.title("♻️ ゴミ分別AIアプリ")
st.write("🖼️ 画像アップロードまたは 📷 カメラ撮影でゴミの種類を判別します。")

# 入力方法の選択
input_method = st.radio("📤 画像の入力方法を選択してください", ["🖼️ 画像アップロード", "📷 カメラ撮影"])
image_file = None

# 入力処理
if input_method == "🖼️ 画像アップロード":
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_file = uploaded_file
elif input_method == "📷 カメラ撮影":
    camera_file = st.camera_input("カメラで撮影してください")
    if camera_file:
        image_file = camera_file

# セッションに画像を保存して再判別を制御
if image_file:
    if "image_file" not in st.session_state or st.session_state.image_file != image_file:
        st.session_state.image_file = image_file
        st.session_state.results_df = None

# AI判別と表示
if "image_file" in st.session_state:
    img = Image.open(st.session_state.image_file)
    st.image(img, use_container_width=True, caption="📸 入力された画像")

    if st.session_state.get("results_df") is None:
        with st.spinner("🤖 AIがゴミを判別中です..."):
            results = model(img)
            st.session_state.results_df = results.pandas().xyxy[0]

    df = st.session_state.results_df

    if df.empty:
        st.warning("⚠️ ゴミが検出されませんでした。別の画像で試してください。")
    else:
        st.subheader("🧠 分別結果")
        for _, row in df.iterrows():
            label = row["name"]
            conf = row["confidence"]
            category = label_to_category.get(label, "⚠️ 未分類のゴミ（手動で確認してください）")
            st.write(f"- **{label}**（信頼度: {conf:.2f}） → {category}")

import streamlit as st
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO

# ────────────
# 1. Hugging Face トークンの取得とログイン
# ────────────
# Streamlit の "Secrets" に保存したトークンを読み込み
hf_token = st.secrets["HUGGINGFACE_TOKEN"]

# Hugging Face 上で認証セッションを開始
login(token=hf_token)

# ────────────
# 2. モデル読み込み関数の定義
# ────────────
@st.cache_resource
def load_yolo_model(repo_id: str, filename: str) -> YOLO:
    """
    YOLO モデルを Hugging Face からダウンロードして読み込む関数。
    @st.cache_resource デコレータを付けることで、
    「一度ダウンロード・初期化したモデル」をキャッシュし、以降は再利用します。

    Args:
        repo_id (str): Hugging Face 上のリポジトリパス（例: "username/model-name"）
        filename (str): リポジトリ内のモデルファイル名（例: "best.pt"）

    Returns:
        YOLO: 初期化済みの YOLO モデルオブジェクト
    """
    # モデルファイルを認証付きでダウンロード
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        use_auth_token=hf_token
    )
    
    # YOLO モデルを初期化
    model = YOLO(model_path)
    return model

# ────────────
# 3. 実際にモデルをロード（初回だけダウンロード／初期化される）
# ────────────
MODEL_REPO = "username/your-trained-yolo-repo"  # ←自分のリポジトリに書き換えてください
MODEL_FILE = "your_model.pt"                   # ←実際のモデルファイル名に書き換えてください

model = load_yolo_model(repo_id=MODEL_REPO, filename=MODEL_FILE)

# ────────────
# 4. 以降のアプリ内で `model` を使って推論できます
# ────────────
# 例: results = model.predict("path/to/image.jpg")




