import streamlit as st
from PIL import Image
from ultralytics import YOLO
import spacy
import tempfile

# タイトル
st.title("ごみ分類アプリ（物体検出 + 自然言語説明）")

# モデルの読み込み
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # 軽量なYOLOv8モデル

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

model = load_model()
nlp = load_spacy_model()

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 画像の読み込みと表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_column_width=True)

    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    # 結果描画
    result_image = results[0].plot()
    st.image(result_image, caption="検出結果", use_column_width=True)

    # 検出されたラベル取得
    labels = results[0].names
    detected_classes = [labels[int(cls)] for cls in results[0].boxes.cls]

    if detected_classes:
        st.subheader("検出されたごみの種類:")
        st.write(", ".join(detected_classes))

        # spaCy を用いた簡単な説明生成
        description = f"The image likely contains: {', '.join(detected_classes)}. These may need to be sorted properly for recycling."
        doc = nlp(description)

        st.subheader("自然言語による説明:")
        st.write(doc.text)
    else:
        st.write("ごみは検出されませんでした。")
