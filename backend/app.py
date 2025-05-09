import streamlit as st
from PIL import Image
import io
from pymongo import MongoClient
import pandas as pd
import datetime

# Import your services directly
from prediction_service import PredictionService
from waste_classification import WasteClassificationService

# Database setup and connection
client = MongoClient("mongodb+srv://easontan7285:nb2XBcriGVzT2QhF@cluster0.u7gxoah.mongodb.net/?tls=true")
db = client["smartbin"]

# Collections
dustbin_col = db["dustbins"]
notification_col = db["notification"]
collect_rubbish_col = db["collectRubbish"]
useraccount_col = db["userAccount"]
rubbish_col = db["rubbish"]

dustbin_df = pd.DataFrame(list(dustbin_col.find()))
notification_df = pd.DataFrame(list(notification_col.find()))
collect_rubbish_df = pd.DataFrame(list(collect_rubbish_col.find()))
useraccount_df = pd.DataFrame(list(useraccount_col.find()))
rubbish_df = pd.DataFrame(list(rubbish_col.find()))

dustbin_hard = "BIN001" 

# Streamlit app title
st.set_page_config(page_title="Smart Waste Management System", layout="wide")
st.title("♻️ Smart Waste Management User Panel")

# Tabs for two main functions
tab1, tab2, tab3 = st.tabs(["Waste Classification", "Collection Order", "Smartbin Installation Order"])

# --- TAB 1: Waste Classification ---
with tab1:
    st.header("🖼 Upload Waste Image for Classification")
    uploaded_file = st.file_uploader("Upload an image of your waste item", type=["jpg", "jpeg", "png","jfif"])

    col1,col2 = st.columns([2, 3])

    if uploaded_file is not None:

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=500)
            
            if st.button("Classify Waste"):
                try:
                    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    prediction_result = PredictionService.predict_image(image, model_name="AlexNet")
                    full_result = WasteClassificationService.process_prediction_result(prediction_result)
                    
                    waste_category = full_result.get("waste_category")
                    disposal_message = full_result.get("disposal_message")
                    disposal_guidelines = full_result.get("disposal_guidelines")

                    with col2:
                        st.success(f"**Waste Type:** {waste_category}")
                        st.info(f"**Disposal Instruction:** {disposal_message}")
                        st.markdown(f"📝 **Guidelines:** {disposal_guidelines}")

                except Exception as e:
                    with col2:
                        st.error(f"Error during classification: {e}")

# --- TAB 2: Collection Order ---
with tab2:
    st.header("📦 Place Collection Order & Earn Reward Points")

    dustbin_id = dustbin_hard
    existing_dustbin = dustbin_col.find_one({'dustbin_id': dustbin_id})

    if existing_dustbin:
        # Retrieve total_reward from userAccount collection
        user_account = useraccount_col.find_one({'dustbin_id': dustbin_id})

        if user_account:
            total_reward = user_account.get('total_reward', 0)  # Default to 0 if field missing
            st.subheader(f"💰 **Total Reward Points (Before Order): RM** {total_reward}")
        else:
            st.warning(f"No user account found")

    with st.form("collection_order_form"):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Address")
        address = st.text_area("Mailing Address")

        submitted = st.form_submit_button("Submit Order")
        
        if submitted:
            try:
                st.write("Received order submission:")
                st.write(f"👤 **Name:** {name}")
                st.write(f"📞 **Phone:** {phone}")
                st.write(f"📧 **Email:** {email}")
                st.write(f"🏠 **Address:** {address}")

                # Static dustbin_id for now; you can later make this dynamic
                location = address  # Using the mailing address as dustbin location
                current_time = datetime.datetime.utcnow()

                # --- Check if dustbin exists ---
                existing_dustbin = dustbin_col.find_one({'dustbin_id': dustbin_id})

                if existing_dustbin:
                    # Update status to "Full"
                    dustbin_col.update_one(
                        {'dustbin_id': dustbin_id},
                        {'$set': {'status': 'Full', 'timeUpdate': current_time, 'location': address}}
                    )

                    useraccount_col.update_one(
                        {'dustbin_id': dustbin_id},
                        {'$set': {
                            'owner_name': name,
                            'location': address,
                            'email': email,
                            'phone': phone
                        }}
                    )

                else:
                    # Create new dustbin entry
                    new_dustbin = {
                        'dustbin_id': dustbin_id,
                        'status': 'Full',
                        'location': address,
                        'timeUpdate': current_time,
                        'type': 'recycle'
                    }
                    dustbin_col.insert_one(new_dustbin)

                    new_user = {
                        'owner_name': name,
                        'dustbin_id': dustbin_id,
                        'location': address,
                        'total_reward': 0,
                        'email': email,
                        'phone': phone
                    }
                    useraccount_col.insert_one(new_user)

                # --- Create notification ---
                new_notification = {
                    'dustbin_id': dustbin_id,
                    'location': address,
                    'timestamp': current_time,
                    'notification_type': 'call',
                    'isCollected': False
                }
                notification_col.insert_one(new_notification)

                st.success(f"✅ Order submitted successfully!")
                st.write(f"📩 **Confirmation Message:** Thank you for supporting sustainable waste management!")

            except Exception as e:
                st.error(f"Error submitting order: {e}")

with tab3:
    st.header("Smartbin Installation Order")

    # Product info
    product_name = "Smart Recycling Dustbin"
    product_description = "This smart dustbin comes with a built-in ultrasonic sensor to detect fullness level and notifies admin when it needs to be collected. Eco-friendly and IoT-integrated."

    # Layout
    col1, col2 = st.columns([1, 2])

    # Product Image
    with col1:
        st.image("images.jpeg", width=250, caption=product_name)

    # Product Details and Quantity Selector
    with col2:
        st.subheader(product_name)
        st.write(product_description)
        st.markdown("**Price:** $120.00")

        # Quantity Control
        if "quantity" not in st.session_state:
            st.session_state.quantity = 1

        col_inc, col_q, col_dec = st.columns([1, 2, 1])

        with col_dec:
            if st.button("➖"):
                if st.session_state.quantity > 1:
                    st.session_state.quantity -= 1

        with col_q:
            st.markdown(f"<h3 style='text-align: center;'>{st.session_state.quantity}</h3>", unsafe_allow_html=True)

        with col_inc:
            if st.button("➕"):
                st.session_state.quantity += 1

        # Add to Cart or Confirm
        if st.button("Add to Order"):
            st.success(f"{st.session_state.quantity} unit(s) of '{product_name}' added to your order.")