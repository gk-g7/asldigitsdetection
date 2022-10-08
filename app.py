import joblib
import numpy as np
import cv2
import mediapipe
import pandas as pd
import streamlit as st
import time
from PIL import Image
import io

dumpedPkl = joblib.load('dumps_pkl.pkl')
colNames = dumpedPkl["df_col_names"]
correlated_features = dumpedPkl["correlated_features"]
corr_scaler = dumpedPkl["corr_scalar"]
uncorr_scaler = dumpedPkl["uncorr_scalar"]
try:
    dumpedPkl["df_col_names"].remove('Label')
except:
    pass

# Page config setup
st.set_page_config(
    page_title="ASL Digits Detection",
    page_icon=":fist:",
    layout="wide",
    menu_items={
        "About": """
        Made by Gokul G
        """,
    },
)

@st.cache(suppress_st_warning=True)
def getModel(model_type, standardize, removeCorrelated):
    modelName = "svmSvc"
    if model_type == "LOGISTIC-REGRESSION":
        modelName = "logReg"
    elif model_type == "KNN":
        modelName = "knn"
    elif model_type == "RANDOM-FOREST":
        modelName = "rbf"

    queryTuple = (modelName, standardize, removeCorrelated)
    print(queryTuple)
    return dumpedPkl[queryTuple]


@st.cache(suppress_st_warning=True)
def predict(file, model_type, standardize, removeCorrelated, flip):
    res = None
    start = time.time()

    with st.spinner("Classifying...."):
        image = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        x = ""
        try:
            finalCoordinates = None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip == True:
                # Flip image around y-axis for correct handedness
                image = cv2.flip(image, 1)
            # using mediapipe hands to get co-ordinates
            mediapipe_hands_model = mediapipe.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                                                    min_detection_confidence=0.7)
            mediapipe_output = mediapipe_hands_model.process(image)
            mediapipe_hands_model.close()
            try:
                mediapipe_output = str(mediapipe_output.multi_hand_landmarks[0])
                mediapipe_output = mediapipe_output.strip().split('\n')
                # removing unwanted details from the mediapipe output
                output_temp = []
                for i in mediapipe_output:
                    if not (i == "landmark {" or i == "}"):
                        output_temp.append(i)
                mediapipe_output = output_temp
                # scrape the coordinate values as list from the mediapipe output string
                coordinates = []
                for i in mediapipe_output:
                    i = i.strip()
                    coordinates.append(i[2:])
                finalCoordinates = coordinates
            except Exception as e:
                print("Exception in mediapipe_image(): %s" % (e))
                raise Exception("Cannot process img")

            for k in finalCoordinates:
                x += str(k)
                x += ','
        except Exception as e:
            print("Exception in pre_process_img(): %s" % (e))
            x = None

        if x is None or x == "":
            return -1, -1, -1

        vals = [i.strip() for i in x[:-1].split(',')]
        df = pd.DataFrame(data=[vals], columns=list(colNames)).astype(float)

        if removeCorrelated and correlated_features is not None:
            df = df.drop(correlated_features, axis=1)

        x = np.array(df.iloc[0])
        x = x.reshape(-1, df.shape[1])

        if standardize:
            if removeCorrelated:
                x = uncorr_scaler.transform(x)
            else:
                x = corr_scaler.transform(x)

        model = getModel(model_type, standardize, removeCorrelated)

        res = int(model.predict(x)[0])
        percent = model.predict_proba(x)[0][res]

    end = time.time()

    # pred_df = pd.DataFrame({"labels": predictions, "confidence": probs})

    return res, percent, round(end-start, 3)


st.markdown(
    """
    <center>
        <h1>
            <b>American Sign Language - Digits Detection</b>
        </h2>
    </center>""",
    unsafe_allow_html=True,
)


with st.expander(label="About the app", expanded=True):
    st.info(
        """
        1. This app predicts the digit based on American Handsign
        2. You can either:
            - Upload an image of a handsign
            - Capture a handsign image
        """
    )

with st.expander(label="Using the app", expanded=True):
    st.write(
        """
        1. Choose the type of prediction form the sidebar: capture or upload
        2. Choose the model: *SVM SVC* or *K-NN* from the sidebar
        3. Click on predict
        """
    )
st.sidebar.image("logo.jpg")


# Type of predicition
input_type = st.sidebar.radio(
    "Type of Input",
    options=["Upload", "Capture"],
    index=0,
    help="Input Type. Choose whether you want to upload an image of handsign or capture a photo",
)

# Choose model
model_type = st.sidebar.radio(
    "Choose Model",
    options=["SVM-SVC", "LOGISTIC-REGRESSION", "KNN", "RANDOM-FOREST"],
    index=0,
    help="Choose which model to be used for prediction SVM SVC or K-NN",
)

standardize = st.sidebar.checkbox("Standardize")
removeCorrelated = st.sidebar.checkbox("Remove Correlated Features")


if input_type == "Upload":

    uploaded_file = st.file_uploader("Choose a Jpeg file", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])

    if uploaded_file is None:
        st.info(
            f"""
            👆 Upload an image
            """
        )
    else:
        col1, col2, col3 = st.columns(3)

        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))

        with col2:
            displayImage = st.image(image)

        s_col1, s_col2, s_col3, s_col4, s_col5, s_col6, s_col7 = st.columns(7)

        with s_col4:
            flip = st.checkbox("Mirror Image")
            if flip:
                displayImage.image(image.transpose(Image.TRANSPOSE.FLIP_LEFT_RIGHT))
            submit_res = st.button("Predict")

        if submit_res:
            res, percent, sec = predict(bytes_data, model_type, standardize, removeCorrelated, flip)
            st.info(f"Processing Time: {sec}s")
            if res != -1:
                pred_print = """
                            <h2>
                                <center>
                                Predicted: {}<br />
                                Confidence: {}%
                                </center>
                            </h2>
                            """
                st.markdown(pred_print.format(res, round(percent * 100, 2)), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <h2>
                        <center>
                            Failed to parse digit. Image is invalid or of low quality or hand sign is not present
                        </center>
                    </h2>
                """, unsafe_allow_html=True)

else:
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is None:
        """
        👆 Capture an image first   
        """
    else:
        bytes_data = img_file_buffer.getvalue()
        col1, col2, col3 = st.columns(3)

        bytes_data = img_file_buffer.read()
        image = Image.open(io.BytesIO(bytes_data))

        with col2:
            displayImage = st.image(image)

        s_col1, s_col2, s_col3, s_col4, s_col5, s_col6, s_col7 = st.columns(7)

        with s_col4:
            flip = st.checkbox("Mirror Image")
            if flip:
                displayImage.image(image.transpose(Image.TRANSPOSE.FLIP_LEFT_RIGHT))
            submit_res = st.button("Predict")
        if submit_res:
            res, percent, sec = predict(bytes_data, model_type, standardize, removeCorrelated, flip)
            st.info(f"Processing Time: {sec}s")
            if res != -1:
                pred_print = """
                                    <h2>
                                        <center>
                                        Predicted: {}<br />
                                        Confidence: {}%
                                        </center>
                                    </h2>
                                    """
                st.markdown(pred_print.format(res, round(percent * 100, 2)), unsafe_allow_html=True)
            else:
                st.markdown("""
                                <h2>
                                    <center>
                                        Failed to parse digit. Image is invalid or of low quality or hand sign is not present
                                    </center>
                                 </h2>
                            """, unsafe_allow_html=True)
