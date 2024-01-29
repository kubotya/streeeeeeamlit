import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

# st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.markdown("オリジナルの画像認識モデルを使って何の画像かを判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                            ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")
        # spinner表示のテスト
        import time
        time.sleep(3) 
        # 予測
        results = predict(img)
        st.write("results =", results)
        # 結果の表示
        st.subheader("判定結果")
        n_top = 10  # 確率が高い順に3位まで返す
        for result in results[:n_top]:
            st.write("result =", result)
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")

        # 円グラフの表示
        pie_labels = [result[1] for result in results[:n_top]]
        pie_probs = [result[2] for result in results[:n_top]]
        fig1, ax1 = plt.subplots()
        wedgeprops = {"width": 0.3, "edgecolor": "white"}
        textprops = {"fontsize": 6}
        ax1.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
                textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
        ax1.set_title("円グラフ")
        st.pyplot(fig1)

        # 棒グラフの表示
        bar_labels = [result[0] for result in results[:n_top]]
        bar_probs = [result[2] for result in results[:n_top]]
        fig2, ax2 = plt.subplots()
        ax2.barh(bar_labels, bar_probs, color='skyblue')
        ax2.set_title("棒グラフ")
        st.pyplot(fig2)

st.sidebar.write("")
st.sidebar.write("")

st.sidebar.markdown("""
このアプリは、「Fashion-MNIST」を訓練データとして使っています。\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
[https://github.com/zalandoresearch/fashion-mnist#license](https://github.com/zalandoresearch/fashion-mnist#license)
""")

