import streamlit as st
from PIL import Image
import torch

# ---------------------------
# 分別カテゴリ（ラベル → 日本語＋アイコン）
# ---------------------------
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

# ---------------------------
# YOLOv5 モデルの読み込み
# ---------------------------
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5n', source='github')

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="ゴミ分別AI", page_icon="♻️")
st.title("♻️ ゴミ分別AIアプリ")
st.write("🖼️ 画像をアップロードまたは 📷 カメラ撮影して、AIがごみの種類を判別します。")

# 画像入力方法の選択
input_method = st.radio("📤 画像の入力方法を選んでください", ["🖼️ アップロード", "📷 カメラ撮影"])
image_file = None

if input_method == "🖼️ アップロード":
    image_file = st.file_uploader("画像ファイルを選択", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("カメラで撮影してください")

# ---------------------------
# 推論処理
# ---------------------------
if image_file is not None:
    img = Image.open(image_file)
    st.image(img, caption="📸 入力画像", use_container_width=True)

    with st.spinner("🤖 ゴミを判別中です..."):
        results = model(img)
        df = results.pandas().xyxy[0]

    if df.empty:
        st.warning("⚠️ ゴミが検出されませんでした。別の画像でお試しください。")
    else:
        st.subheader("🧠 分別結果")
        for _, row in df.iterrows():
            label = row["name"]
            conf = row["confidence"]
            category = label_to_category.get(label, "⚠️ 未分類のゴミ（手動で確認してください）")
            st.write(f"- **{label}**（信頼度: {conf:.2f}） → {category}")
