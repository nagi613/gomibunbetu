import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO
import pandas as pd

# ラベル → ゴミカテゴリ（絵文字付き）
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

# ──────────────────────────────
# 1. Hugging Face 認証 & モデル準備
# ──────────────────────────────

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

@st.cache_resource
def load_yolo_model(repo_id: str, filename: str) -> YOLO:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=hf_token)
    return YOLO(model_path)

# ※ ここは自分のモデルリポジトリに置き換えてください
MODEL_REPO = "your-username/your-yolo-repo"  # 例: "takashi/gomi-detector"
MODEL_FILE = "best.pt"

model = load_yolo_model(MODEL_REPO, MODEL_FILE)

# ──────────────────────────────
# 2. Streamlit UI 部分
# ──────────────────────────────

st.set_page_config(page_title="ゴミ分別AI", page_icon="♻️", layout="centered")
st.title("♻️ ゴミ分別AIアプリ")
st.write("🖼️ 画像アップロードまたは 📷 カメラ撮影でゴミの種類を判別します。")

input_method = st.radio("📤 画像の入力方法を選択してください", ["🖼️ 画像アップロード", "📷 カメラ撮影"])

# 入力欄をラップして、UIのエラー防止
input_area = st.empty()
image_file = None

with input_area:
    if input_method == "🖼️ 画像アップロード":
        uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_file = uploaded_file
    else:
        camera_file = st.camera_input("カメラで撮影してください")
        if camera_file:
            image_file = camera_file

# ──────────────────────────────
# 3. 推論処理
# ──────────────────────────────

if image_file:
    img = Image.open(image_file)
    st.image(img, use_container_width=True, caption="📸 入力画像")

    with st.spinner("🤖 ゴミを判別中です..."):
        results = model.predict(img)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.warning("⚠️ ゴミが検出されませんでした。別の画像でお試しください。")
        else:
            st.subheader("🧠 分別結果")

            for box in boxes:
                label_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[label_id]
                category = label_to_category.get(label, "⚠️ 未分類（手動確認）")

                st.write(f"- **{label}**（信頼度: {conf:.2f}）→ {category}")
