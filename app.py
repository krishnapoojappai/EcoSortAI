import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog


st.set_page_config(
    page_title="EcoSortAI",
    page_icon="♻️",
    layout="wide"
)

st.markdown("""
<style>

/* Main spacing */
.main {
    padding-top: 1rem;
}

/* Center title */
h1 {
    text-align: center;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="metric-container"] * {
    color: white !important;
}

/* Bigger Tabs */
button[data-baseweb="tab"] {
    font-size: 20px !important;
    font-weight: bold !important;
    padding: 15px 35px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


with open("model/waste_classifier.pkl", "rb") as f:
    model = pickle.load(f)


with st.sidebar:

    st.header("♻️ About EcoSortAI")

    st.write("""
EcoSortAI is an AI-powered waste classification system that helps users:

• Classify waste materials

• Learn proper disposal methods

• Follow Reduce, Reuse and Recycle principles

• Promote environmental sustainability
""")

    st.divider()

    st.subheader("🧠 Model Information")

    st.write("**Model:** Support Vector Machine (SVM)")
    st.write("**Features:** HOG Features")
    st.write("**Dataset:** TrashNet")
    st.write("**Classes:** 6")


st.title("♻️ EcoSortAI")


st.markdown("""
<div style="
background: linear-gradient(90deg,#14532D,#166534,#15803D);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
margin-bottom:20px;
">

<h2>🌍 Smart Waste Classification for a Sustainable Future</h2>

<p>
Upload an image and let AI identify waste, recommend disposal methods,
and encourage sustainable practices through Reduce, Reuse and Recycle.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
### AI-Powered Waste Classification & Sustainability Assistant

Transform waste into opportunity.

Upload an image and EcoSortAI will:

- 🧠 Classify the waste material into CARDBOARD / GLASS / METAL / PAPER / PLASTIC / TRASH
- ♻️ Suggest proper disposal methods
- 🌱 Recommend sustainable actions
- 🌍 Promote responsible waste management
""")


uploaded_file = st.file_uploader(
    "Upload a waste image [must be in white background, bright lighting and object must be clearly focused]",
    type=["jpg", "jpeg", "png"]
)


icons = {
    "plastic": "🧴",
    "paper": "📰",
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "trash": "🗑️"
}


recommendations = {

    "plastic": {
        "dispose": "Place in a plastic recycling bin.",
        "reduce": "Use reusable bottles and shopping bags.",
        "reuse": "Convert bottles into planters or storage containers.",
        "recycle": "Send to a plastic recycling facility.",
        "impact": "Plastic can take hundreds of years to decompose and contributes significantly to environmental pollution."
    },

    "paper": {
        "dispose": "Place in a paper recycling bin.",
        "reduce": "Use digital notes and documents whenever possible.",
        "reuse": "Use for rough work, wrapping or crafts.",
        "recycle": "Send to a paper recycling center.",
        "impact": "Recycling paper helps reduce deforestation and saves water and energy."
    },

    "cardboard": {
        "dispose": "Flatten and place in cardboard recycling.",
        "reduce": "Choose products with minimal packaging.",
        "reuse": "Reuse as storage boxes or DIY projects.",
        "recycle": "Send to a cardboard recycling facility.",
        "impact": "Cardboard is highly recyclable and helps reduce landfill waste."
    },

    "glass": {
        "dispose": "Place in a glass recycling bin.",
        "reduce": "Use refillable containers.",
        "reuse": "Reuse jars and bottles for storage.",
        "recycle": "Send to a glass recycling center.",
        "impact": "Glass can be recycled indefinitely without losing quality."
    },

    "metal": {
        "dispose": "Place in a metal recycling bin.",
        "reduce": "Avoid unnecessary canned products.",
        "reuse": "Use containers for DIY projects.",
        "recycle": "Send to a metal recycling facility.",
        "impact": "Recycling metal saves energy and reduces mining activities."
    },

    "trash": {
        "dispose": "Dispose in the general waste bin.",
        "reduce": "Reduce usage of non-recyclable products.",
        "reuse": "Check if any component can be repurposed.",
        "recycle": "Follow local waste management guidelines.",
        "impact": "Non-recyclable waste increases landfill usage and environmental burden."
    }
}


if uploaded_file:

    try:

        image = Image.open(uploaded_file)

        
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        img = np.array(image)

    
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGBA2RGB
            )

        IMG_SIZE = 128

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        confidence = None

        try:
            probabilities = model.predict_proba(features)
            confidence = np.max(probabilities) * 100

        except Exception:
            confidence = None

        if confidence is not None:

            if confidence < 40:
                st.error(f"❌ Unknown or unsupported waste type.\n\nConfidence: {confidence:.2f}%")

                st.stop()

            elif confidence < 60:
                st.warning(f"⚠️ Low confidence prediction ({confidence:.2f}%). Results may be inaccurate."
        )

            else:
                st.success( f"✅ High confidence prediction ({confidence:.2f}%).")

        with col2:

            st.markdown(
            f"""
            <div style="
                background-color:#1E3A8A;
                color:white;
                padding:15px;
                border-radius:10px;
                font-size:22px;
                font-weight:bold;
                text-align:center;
                margin-bottom:20px;
            ">
                {icons[prediction]} Detected Waste: {prediction.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Waste Type",
                prediction.upper()
            )

        with metric_col2:
            if confidence is not None:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

        info = recommendations[prediction]

        st.divider()

        st.subheader("🌍 Environmental Impact")

        st.warning(info["impact"])

        st.subheader("♻️ Sustainability Recommendations")

        tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🗑️ Disposal",
            "🌱 Reduce",
            "🔄 Reuse",
            "♻️ Recycle"
        ]
    )

        with tab1:
            st.info(info["dispose"])

        with tab2:
            st.success(info["reduce"])

        with tab3:
            st.warning(info["reuse"])

        with tab4:
            st.error(info["recycle"])

        st.success(
        "Thank you for contributing to a cleaner and more sustainable future!"
    )
    
    except Exception as e:

     st.error(
            "❌ Unable to process the image. Please upload a valid image."
        )


st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        EcoSortAI • Built using Streamlit, OpenCV, Scikit-Learn, HOG Features and SVM
    </div>
    """,
    unsafe_allow_html=True
)
