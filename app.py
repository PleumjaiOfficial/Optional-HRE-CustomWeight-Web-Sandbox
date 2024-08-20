import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# """
# AI 
# """

class distance_on_performance():

    evaluation_matrix = np.array([])    # Matrix
    weighted_normalized = np.array([])  # Weight matrix
    normalized_decision = np.array([])  # Normalisation matrix

    M = 0  # Number of rows/employee
    N = 0  # Number of columns/criteria

    def __init__(self, evaluation_matrix, weight_matrix, criteria):
        # M×N matrix
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.evaluation_matrix)

        # N attributes/criteria
        self.column_size = len(self.evaluation_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix/sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    def step_2(self):
        # normalized scores
        self.normalized_decision = np.copy(self.evaluation_matrix)
        sqrd_sum = np.zeros(self.column_size)
        for i in range(self.row_size):
            for j in range(self.column_size):
                sqrd_sum[j] += self.evaluation_matrix[i, j]**2
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.normalized_decision[i,
                                         j] = self.evaluation_matrix[i, j]/(sqrd_sum[j]**0.5)

    def step_3(self):
        from pdb import set_trace
        self.weighted_normalized = np.copy(self.normalized_decision)
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.weighted_normalized[i, j] *= self.weight_matrix[j]

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = max(self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = min(self.weighted_normalized[:, i])

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = np.copy(self.weighted_normalized)
        self.best_distance_mat = np.copy(self.weighted_normalized)

        for i in range(self.row_size):
            for j in range(self.column_size):
                self.worst_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.worst_alternatives[j])**2
                self.best_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.best_alternatives[j])**2

                self.worst_distance[i] += self.worst_distance_mat[i][j]
                self.best_distance[i] += self.best_distance_mat[i][j]

        for i in range(self.row_size):
            self.worst_distance[i] = self.worst_distance[i]**0.5
            self.best_distance[i] = self.best_distance[i]**0.5

    def step_6(self):
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # calculate the similarity to the worst condition
            self.worst_similarity[i] = self.worst_distance[i] / (self.worst_distance[i] + self.best_distance[i])

            # calculate the similarity to the best condition
            self.best_similarity[i] = self.best_distance[i] / (self.worst_distance[i] + self.best_distance[i])

    def ranking(self, data):
        return [i+1 for i in data.argsort()]

    def rank_to_worst_similarity(self):
        # return rankdata(self.worst_similarity, method="min").astype(int)
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        # return rankdata(self.best_similarity, method='min').astype(int)
        return self.ranking(self.best_similarity)

    def calc(self):
        # st.write("Step 1: Raw data")
        # st.write(self.evaluation_matrix)
        self.step_2()
        # st.write("Step 2: Normalized Matrix")
        # st.write(self.normalized_decision)
        self.step_3()
        # st.write("Step 3: Calculate the weighted normalized decision matrix")
        # st.write(self.weighted_normalized)
        self.step_4()
        # st.write("Step 4: Determine the worst alternative and the best alternative")
        # st.write(f"Worst: {self.worst_alternatives}")
        # st.write(f"Best: {self.best_alternatives}")
        self.step_5()
        # st.write("Step 5: Distance from Best to Worst")
        # st.write(f"Worst distance: {self.worst_distance}")
        # st.write(f"Best distance: {self.best_distance}")
        self.step_6()
        # st.write("Step 6: Similarity")
        # st.write(f"Worst similarity: {self.worst_similarity}")
        # st.write(f"Best similarity: {self.best_similarity}")


def tier_rank(tier_labal, quantile, target_columns):
    results, bin_edges = pd.qcut(target_columns,
                            q= quantile,
                            labels= tier_labal,
                            retbins=True)

    tier_table = pd.DataFrame(zip(bin_edges, tier_labal),
                                columns=['Threshold', 'Tier'])
    
    return tier_table

def adjust_score(_max, _min, in_array):

    input = in_array.copy()
    result = []

    for i in input:
        peer = (i - min(input)) / max(input)
        z = peer * (_max - _min) + _min
        result.append(z)

    return result

def normalization_pms_toptalent(df, score_columns, weight_matrix, criteria):
    ''' 
    -- weight_matrix : list of weights corresponding to score_columns
    -- criteria : list of boolean values, True for criteria where higher is better, False otherwise
    '''
    
    # Extract the relevant quality scores from the DataFrame based on the provided columns
    quality = df[score_columns].values

    # Ensure the weights are normalized
    weight_matrix = np.array(weight_matrix, dtype="float")
    weight_matrix = weight_matrix / sum(weight_matrix)

    # Initialize the distance_on_performance class with dynamic score columns, weights, and criteria
    top = distance_on_performance(quality, weight_matrix, criteria)
    top.calc()

    members = df[['PERSON_ID']].values.flatten()

    raw = top.best_similarity
    _result = 1 - raw

    _max = 1
    _min = 0

    result = adjust_score(_max, _min, _result)

    arr = [*sorted(zip(members, result), key=lambda x: x[1], reverse=True)]

    reg_df = pd.DataFrame(arr, columns=['PERSON_ID', 'TALENT_SCORE'])

    return reg_df

def toptalent(df, score_columns, weight_matrix, criteria):
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df[score_columns])

    scaled_df = pd.DataFrame(scaled_data, columns=[col + '_SCALED' for col in score_columns])

    final_selection_df = pd.concat([df, scaled_df], axis=1)

    reg_df = normalization_pms_toptalent(final_selection_df, [col + '_SCALED' for col in score_columns], weight_matrix, criteria)

    bin_labels = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    q = [0, .4, .6, .8, .9, .1]

    tier_table = tier_rank(bin_labels, q, reg_df['TALENT_SCORE'])
    tier_table['Percentile'] = q[:-1]

    bins = list(tier_table['Threshold']) + [float('inf')]
    labels = tier_table['Tier']

    reg_df['TIER'] = pd.cut(reg_df['TALENT_SCORE'], bins=bins, labels=labels, right=False)

    toptalent = reg_df.copy(deep=True)
    tier_order = ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze']
    toptalent['TIER'] = pd.Categorical(toptalent['TIER'], categories=tier_order, ordered=True)
    toptalent = toptalent.sort_values('TIER')
    tier_table = tier_table.sort_values('Threshold', ascending=False)

    return toptalent, tier_table


# """
# APPICATION
# """

# Define job families and sources
sources = ["Direct Manager", "Other Manager", "Peer within Function", "Self", "External", "Peer across Department", "Subordinate"]
scores_columns = ['CQ1', 'CQ2', 'CQ3', 'CQ4', 'DQ1', 'DQ2', 'DQ3', 'DQ4', 'EQ1', 'EQ2', 'EQ3', 'EQ4']
columns = ['PERSON_ID', 'SOURCE'] + scores_columns

# Initialize the DataFrame structure
data = pd.DataFrame(columns=columns)

# Streamlit app title
st.title("Employee Score Input Simulation")
# st.subheader("How many employees do you want to compare?")
# num_emp = st.number_input("Number of employees", min_value=1, max_value=100, value=3)

# Define expected columns
expected_columns = ['PERSON_ID', 'SOURCE', 'CQ1', 'CQ2', 'CQ3', 'CQ4', 'DQ1', 'DQ2', 'DQ3', 'DQ4', 'EQ1', 'EQ2', 'EQ3', 'EQ4']

# Step 1: Upload XLSX file
st.header("Step 1: Upload Employee Scores File")
uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx")

if uploaded_file:
    # Load the uploaded file into a DataFrame
    try:
        data = pd.read_excel(uploaded_file)
        
        # Check if the uploaded file has the required columns
        if list(data.columns) == expected_columns:
            st.success("File loaded successfully!")
            st.write("Loaded Data:")
            st.dataframe(data)
        else:
            st.error(f"Incorrect template. Please ensure the file has the following columns: {expected_columns}")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Display the resulting DataFrame
# st.write("Resulting DataFrame:")
# st.dataframe(data)

# Line break
st.markdown("***")

# Initialize dictionary for weights
base_weights = {}

# Input fields for each source weight using columns
st.header("Step 2: Input Weights for Each Source")
weight_cols = st.columns(len(sources))
for i, source in enumerate(sources):
    with weight_cols[i]:
        base_weights[source] = st.number_input(f"{source}", min_value=0.0, max_value=1.0, value=0.1)

# Display the weights
st.write("Base Weights Dictionary:")
st.json(base_weights)

# Line break
st.markdown("***")

if not data.empty:
    weighted_data = data.copy()
    # Step 3: Apply the weights
    st.header("Step 3: Apply Weights and View Weighted Scores")

    for source in sources:
        weight = base_weights[source]
        weighted_data.loc[weighted_data['SOURCE'] == source, scores_columns] *= weight

    st.write("Weighted DataFrame:")
    st.dataframe(weighted_data)

    # Optionally, sum up the weighted scores for each employee
    sum_scores = weighted_data.groupby("PERSON_ID")[scores_columns].sum().reset_index()
    st.write("Sum of Weighted Scores by Employee:")
    st.dataframe(sum_scores)

    # Sum the scores for each category
    sum_scores['TOTAL_C'] = sum_scores[['CQ1', 'CQ2', 'CQ3', 'CQ4']].sum(axis=1)
    sum_scores['TOTAL_D'] = sum_scores[['DQ1', 'DQ2', 'DQ3', 'DQ4']].sum(axis=1)
    sum_scores['TOTAL_E'] = sum_scores[['EQ1', 'EQ2', 'EQ3', 'EQ4']].sum(axis=1)

    # Display the DataFrame with summed scores
    st.write("Sum of Total Scores by Employee:")
    st.dataframe(sum_scores)

    # Line break
    st.markdown("***")
    st.header("Step 4: Evaluate the toptalent")
    # Display the DataFrame with summed scores
    st.write("Sum of Total Scores by Employee:")
    st.dataframe(sum_scores[['PERSON_ID', 'TOTAL_C', 'TOTAL_D', 'TOTAL_E']])

    criteria = [True] * 3  # 7 sources
    weighted_total_columns = [100, 100, 100]
    total_columns = ['TOTAL_C', 'TOTAL_D', 'TOTAL_E']

    if st.button("Calculate Top Talent"):
        with st.spinner("Calculating..."):
            st_output = st.empty()  # Placeholder to capture print outputs
            # result_df, tier_table = toptalent_dynamic_weight(data, criteria, source_columns, total_columns, base_weights)
            result_df, tier_table =  toptalent(sum_scores, ['TOTAL_C', 'TOTAL_D', 'TOTAL_E'], [100, 100, 100], [True, True, True])
        
        st.success("Calculation Complete!")
        st.subheader("Top Talent Result")
        st.dataframe(result_df)
        st.subheader("Tier Table")
        st.dataframe(tier_table)


        st.subheader("Optional: Top talent by HR")
        toptalent_list_by_hr = [336, 321, 327, 7079, 7097, 229, 6713, 6744, 6271, 8039, 235]
        filtered_df = result_df[result_df['PERSON_ID'].isin(toptalent_list_by_hr)].sort_values(by='TALENT_SCORE', ascending=False)

        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            # Skip or handle the case where no matching PERSON_IDs are found
            st.write("No matching top talent by HR found.")
else:
    st.header("🗳️ Please, download template data first")
    st.write("and then have a cup of tea while waiting for the calculation...")

    # File path to the file you want to offer for download
    file_path = "template_scores.xlsx"

    # Open the file in binary mode
    with open(file_path, "rb") as file:
        st.download_button(
            label="Download the Template data",
            data=file,
            file_name="template_scores.xlsx",  # Set the desired file name for download
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # MIME type for XLSX files
        )