import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Load the dataset
try:
    data_path = 'ubereats.csv'  # Adjust the path as necessary
    ubereats_data = pd.read_csv(data_path)
    ubereats_data['date'] = pd.to_datetime(ubereats_data['date'])  # Ensure date is in datetime format
except FileNotFoundError:
    st.error("Dataset not found. Please check the file path.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("Dataset is empty. Please provide a valid CSV file.")
    st.stop()

# Introduction to the dataset
def introduction():
    st.markdown("""
    ## Welcome to the UberEats Data Analysis Assistant!
    This app provides insights into customer feedback from UberEats.
    The dataset contains tweets categorized into various segments to understand service quality.
    Segments include:
    - **Other**: Tweets that are not complaints.
    - **Customer Service**: Feedback related to customer support.
    - **Fees or Charges**: Complaints about pricing.
    - **Driver**: Issues related to drivers.
    - **Wrong Delivery**: Complaints about incorrect orders.
    - **Late Delivery**: Issues with delayed deliveries.
    - **Cancelled Order**: Complaints about order cancellations.
    - **Non-Complaint**: Tweets that are not complaints.
    Ask me any questions about the data, and I'll do my best to provide answers!
    Type 'terminate' to stop the analysis at any time.
    """)

# Function to process and respond to questions
def process_question(question, num_answers, location, month):
    if ubereats_data is None:
        return "No data available to process your question."

    # Apply location and month filters
    filtered_data = ubereats_data
    if location != "All":
        filtered_data = filtered_data[filtered_data['UserLoc'] == location]
    if month != "All":
        filtered_data = filtered_data[filtered_data['date'].dt.strftime('%B') == month]

    if 'main segments' in question or 'different segments' in question:
        return filtered_data['segment'].value_counts().to_dict()
    elif 'overall sentiment' in question:
        return filtered_data['sentiment'].value_counts().to_dict()
    elif 'common complaints' in question or 'most common complaints' in question:
        return filtered_data[filtered_data['segment'] != 'non-complaint']['content'].tolist()[:num_answers]
    elif 'praise' in question or 'positive feedback' in question or 'common positive comments' in question:
        return filtered_data[filtered_data['segment'] == 'non-complaint']['content'].tolist()[:num_answers]
    elif 'customer service' in question:
        return filtered_data[filtered_data['segment'] == 'Customer service']['content'].tolist()[:num_answers]
    elif 'fees' in question or 'charges' in question or 'opinions on UberEats pricing' in question:
        return filtered_data[filtered_data['segment'] == 'Fees or charges']['content'].tolist()[:num_answers]
    elif 'driver' in question:
        return filtered_data[filtered_data['segment'] == 'Driver']['content'].tolist()[:num_answers]
    elif 'wrong delivery' in question:
        return filtered_data[filtered_data['segment'] == 'Wrong delivery']['content'].tolist()[:num_answers]
    elif 'late delivery' in question:
        return filtered_data[filtered_data['segment'] == 'Late delivery']['content'].tolist()[:num_answers]
    elif 'cancelled order' in question or 'order cancellations' in question:
        return filtered_data[filtered_data['segment'] == 'Cancelled order']['content'].tolist()[:num_answers]
    elif 'sentiment distribution' in question or 'sentiment vary by segment' in question:
        return filtered_data['sentiment'].value_counts().to_dict()
    elif 'negative sentiment' in question:
        return filtered_data[filtered_data['sentiment'] == 'negative']['segment'].value_counts().to_dict()
    elif 'positive sentiment' in question:
        return filtered_data[filtered_data['sentiment'] == 'positive']['segment'].value_counts().to_dict()
    elif 'sentiment vary' in question or 'how does sentiment vary by location' in question:
        return filtered_data.groupby('segment')['sentiment'].value_counts(normalize=True).to_dict()
    elif 'feedback changed' in question or 'over time' in question:
        return filtered_data.groupby(pd.to_datetime(filtered_data['date']).dt.to_period('M'))['content'].count().to_dict()
    elif 'complaints vary' in question or 'across regions' in question:
        return filtered_data.groupby('UserLoc')['segment'].value_counts().to_dict()
    elif 'customer service be improved' in question:
        return ["Improving response time", "Training staff better", "Offering refunds or discounts"]
    elif 'pricing' in question or 'fees be made more transparent' in question:
        return ["Offering transparent pricing", "Providing clear breakdowns of costs"]
    elif 'driver issues' in question or 'about drivers' in question or 'driver-related issues' in question:
        return ["Background checks", "Training programs", "Customer feedback systems"]
    elif 'wrong delivery reasons' in question:
        return ["Order mix-ups", "Incorrect address", "Miscommunication"]
    elif 'delivery accuracy be improved' in question:
        return ["Better tracking systems", "Verification processes", "Driver training"]
    elif 'causes of late deliveries' in question:
        return ["Traffic delays", "Order processing issues", "Driver availability"]
    elif 'delivery times be improved' in question:
        return ["Real-time tracking", "Optimizing delivery routes", "Ensuring driver availability"]
    elif 'order reliability be improved' in question:
        return ["Better inventory management", "Reliable delivery systems", "Clear communication"]
    elif 'customer complaints' in question or 'order cancellations' in question:
        return ["Order not being prepared", "Driver not showing up", "Technical issues"]
    elif 'compare to other services' in question:
        return ["UberEats has more complaints about delivery times", "Competitors might have better customer service"]
    elif 'standout positive reviews' in question:
        return filtered_data[filtered_data['segment'] == 'non-complaint']['content'].tolist()[:num_answers]
    elif 'tweets in Other segment' in question:
        return filtered_data[filtered_data['segment'] == 'other']['content'].tolist()[:num_answers]
    elif 'highly engaged tweets' in question or 'highly engaged tweets with significant insights' in question:
        return filtered_data.sort_values(by='engagement rate', ascending=False).head(num_answers)['content'].tolist()
    else:
        return "I didn't understand your question. Please ask another question."

# Function to get response based on question
def get_response(question, num_answers, location, month):
    possible_answers = [
        "Tell me about the different segments in the data.",
        "What are the common issues with customer service?",
        "What do customers think about fees or charges?",
        "What are the main complaints about drivers?",
        "What issues are related to wrong deliveries?",
        "What are the causes of late deliveries?",
        "Why do customers complain about order cancellations?",
        "What are the common positive comments?",
        "What kinds of tweets are in the 'Other' segment?",
        "How does UberEats compare to other services?",
        "What insights can competitors use to improve their services?",
        "How engaged are customers with tweets about UberEats?",
        "Are there any highly engaged tweets with significant insights?",
        "What trends are in the 'Other' segment?",
        "How can customer service be improved?",
        "What feedback is there about customer service improvements?",
        "What are the opinions on UberEats pricing?",
        "How can fees be made more transparent?",
        "How can driver-related issues be addressed?",
        "What positive feedback is there about drivers?",
        "How can order reliability be enhanced?",
        "What are the common issues with drivers?",
        "How does sentiment vary by location?",
        "What feedback is there about pricing?",
        "What are the positive aspects mentioned?",
        "What do customers appreciate about UberEats?",
        "What are the issues with customer service?",
        "What feedback is there about fees?",
        "What are the common issues mentioned in tweets?",
        "What is the overall sentiment in the data?",
        "What are the main segments in the data?",
        "What are the most common complaints?",
        "What praise or positive feedback is in the data?",
        "What are the issues related to customer service, fees, drivers, wrong deliveries, late deliveries, and cancelled orders?",
        "How does sentiment vary by segment?",
        "How has feedback changed over time?",
        "How do complaints vary across regions?",
        "How can customer service, pricing, driver issues, delivery accuracy, and order reliability be improved?",
        "How can customer service be improved?",
        "How can fees be made more transparent?",
        "How can driver-related issues be addressed?",
        "What are the common issues with wrong deliveries, late deliveries, and order cancellations?",
        "How does UberEats compare to other services?"
    ]

    if question is None:
        return "Please ask a question about the UberEats data."

    # Use the SentenceTransformer model for semantic similarity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(possible_answers)
    question_embedding = model.encode(question)
    similarities = util.pytorch_cos_sim(question_embedding, embeddings)

    # Get the most similar question
    max_similarity_index = similarities.argmax().item()
    matched_question = possible_answers[max_similarity_index]

    # Process the matched question
    return process_question(matched_question, num_answers, location, month)

# Function to display the chart
def display_chart(data, title):
    st.write(f"### {title}")
    fig, ax = plt.subplots()
    pd.Series(data).plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Main function
def main():
    st.set_page_config(page_title="UberEats Data Analysis", layout="wide")
    st.sidebar.header("Filters")

    # Location filter
    locations = ['All'] + ubereats_data['UserLoc'].unique().tolist()
    selected_location = st.sidebar.selectbox("Select Location", locations, index=0)

    # Month filter
    months = ['All'] + ubereats_data['date'].dt.strftime('%B').unique().tolist()
    selected_month = st.sidebar.selectbox("Select Month", months, index=0)

    # Number of answers slider
    num_answers = st.sidebar.slider("Select number of answers to display", 1, 20, 5)

    # Call the introduction function
    introduction()

    # Adding placeholder text to guide users
    question = st.text_input("Type your question here:", placeholder="e.g., What are the common issues with customer service?")

    if st.button("Submit"):
        if question.strip() != '':
            answer = get_response(question, num_answers, selected_location, selected_month)
            if isinstance(answer, dict):
                display_chart(answer, "Answer")
            elif isinstance(answer, list):  # Check if the answer is a list
                st.markdown("**Here are some examples:**")
                answer_text = "\n\n".join(answer)  # Join list elements into a single string
                st.markdown(answer_text.replace("\n\n", "\n\n---\n\n"))  # Add a line to separate tweets for better readability
            else:
                st.write(answer)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
