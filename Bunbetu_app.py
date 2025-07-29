import streamlit as st
from PIL import Image
import torch

# --- ラベル → カテゴリ変換 ---
label_to_category = {import streamlit as st
from PIL import Image
import torch

# カテゴリ対応表
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

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5n', source='github')

model = load_model()

st.set_page_config(page_title="ゴミ分別AI", page_icon="♻️")
st.title("♻️ ゴミ分別AIアプリ")
st.write("🖼️ 画像アップロードでゴミの種類を自動判別します。")

img_data = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if img_data:
    try:
        img = Image.open(img_data)
        st.image(img, caption="📷 入力画像", use_container_width=True)

        with st.spinner("AIがごみを判別中..."):
            results = model(img)
            df = results.pandas().xyxy[0]

        if df.empty:
            st.warning("⚠️ ゴミが検出されませんでした。別の画像でお試しください。")
        else:
            st.subheader("🧠 分別結果")
            for _, row in df.iterrows():
                label = row["name"]
                conf = row["confidence"]
                category = label_to_category.get(label, "⚠️ 未分類（手動確認）")
                st.write(f"- **{label}**（信頼度: {conf:.2f}） → {category}")
    except Exception as e:
        st.error(f"エラー: {e}")

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

# --- モデル読み込み ---
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5n', source='github')

model = load_model()

# --- UI構築 ---
st.set_page_config(page_title="ゴミ分別AI", page_icon="♻️")
st.title("♻️ ゴミ分別AIアプリ")
st.write("📷 カメラまたは 🖼️ アップロード画像でAIがごみを自動判別します。")

# --- 入力方式選択 ---
method = st.radio("画像の入力方法を選択してください", ["🖼️ 画像アップロード", "📷 カメラ撮影"])

img_data = None
if method == "🖼️ 画像アップロード":
    img_data = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
else:
    img_data = st.camera_input("カメラで撮影してください")

# --- 推論処理 ---
if img_data:
    try:
        image = Image.open(img_data)
        st.image(image, caption="📸 入力画像", use_container_width=True)

        with st.spinner("AIがごみを識別中..."):
            results = model(image)
            df = results.pandas().xyxy[0]

        if df.empty:
            st.warning("⚠️ ゴミが検出されませんでした。別の画像をお試しください。")
        else:
            st.subheader("🧠 分別結果")
            for _, row in df.iterrows():
                label = row["name"]
                conf = row["confidence"]
                category = label_to_category.get(label, "⚠️ 未分類（手動確認）")
                st.write(f"- **{label}**（信頼度: {conf:.2f}） → {category}")

    except Exception as e:
        st.error(f"画像処理中にエラーが発生しました: {e}")
