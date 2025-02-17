import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the CSV files
result_csv_path = "result.csv"
result2_csv_path = "result2.csv"

df = pd.read_csv(result_csv_path)
makeup = pd.read_csv(result2_csv_path)

# Preprocess Skincare Data
df['concern 2'].fillna('', inplace=True)
df['concern 3'].fillna('', inplace=True)
df['concern'] = df['concern'] + ',' + df['concern 2'] + ',' + df['concern 3']
df.drop(columns=['concern 2', 'concern 3', 'spf', 'key ingredient', 'formulation'], inplace=True)

df2 = df[df['label'].isin(['face-moisturisers', 'mask-and-peel', 'cleanser', 'eye-cream'])]
df2 = df2[df2['skin type'].notna()]
df2['concern'] = df2['concern'].str.lower()
df2['brand'] = df2['brand'].str.lower()
df2['name'] = df2['name'].str.lower()
df2['skin type'] = df2['skin type'].str.lower()
df2['concern'] = df2['concern'].str.replace(' and ', ',').str.replace(' or ', ',')

top_concerns = {
    'face-moisturisers': 'general care',
    'mask-and-peel': 'daily use',
    'cleanser': 'general care',
    'eye-cream': 'fine lines,wrinkles,dark circles,eye bags'
}
for i in range(len(df2)):
    if pd.isnull(df2.iloc[i]['concern']):
        label = df2.iloc[i]['label']
        df2.at[i, 'concern'] = top_concerns[label]

# Define features and encode
features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
            'fine lines', 'wrinkles', 'redness', 'dull', 'pore', 'pigmentation',
            'blackheads', 'whiteheads', 'blemishes', 'dark circles', 'eye bags', 'dark spots']

entries = len(df2)
one_hot_encodings = np.zeros((entries, len(features)))

for i in range(entries):
    skin_type = df2.iloc[i]['skin type']
    if skin_type == 'all':
        one_hot_encodings[i][:5] = 1
    elif skin_type in features:
        one_hot_encodings[i][features.index(skin_type)] = 1

def search_concern(target, i):
    concern_value = str(df2.iloc[i]['concern']) if pd.notna(df2.iloc[i]['concern']) else ''
    return target in concern_value

for i in range(entries):
    for j in range(5, len(features)):
        if search_concern(features[j], i):
            one_hot_encodings[i][j] = 1

# Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(one_hot_encodings)
distances, indices = nbrs.kneighbors(one_hot_encodings)

def name2index(name):
    return df2[df2['name'] == name].index[0]

def index2prod(index):
    return df2.iloc[index]

def wrap(info_arr):
    return {
        'brand': info_arr[0],
        'name': info_arr[1],
        'price': info_arr[2],
        'url': info_arr[3],
        'skin type': info_arr[4],
        'concern': str(info_arr[5]).split(',')
    }

def recs_cs(vector=None, name=None, label=None, count=5):
    products = []
    if name:
        idx = name2index(name)
        fv = one_hot_encodings[idx]
    elif vector is not None:
        fv = vector
    else:
        return []

    cs_values = cosine_similarity(np.array([fv]), one_hot_encodings)[0]
    df2['cs'] = cs_values

    dff = df2[df2['label'] == label] if label else df2
    if name:
        dff = dff[dff['name'] != name]
    recommendations = dff.sort_values('cs', ascending=False).head(count)
    data = recommendations[['brand', 'name', 'price', 'url', 'skin type', 'concern']].to_dict('split')['data']
    for element in data:
        products.append(wrap(element))
    return products

def recs_essentials(vector=None, name=None):
    response = {}
    for label in df2['label'].unique():
        response[label] = recs_cs(vector=vector, name=name, label=label)
    return response

if __name__ == "__main__":
    print("Welcome to the Skincare Recommendation Engine!")
    print("Please provide the following details to get recommendations.\n")

    # Collect user inputs
    skin_type = input("Enter your skin type (e.g., Normal, Dry, Oily, Combination, Sensitive): ").lower().strip()
    concerns = input("Enter your skin concerns (comma-separated, e.g., Acne, Dark Spots, Fine Lines): ").lower().split(',')

    # Create input vector
    input_vector = [1 if feature in skin_type or feature in concerns else 0 for feature in features]

    # Generate and display recommendations
    recommendations = recs_essentials(vector=input_vector, name=None)
    # Generate and display recommendations
    recommendations = recs_essentials(vector=input_vector, name=None)
    print("\nRecommended Products Based on Your Input:")
    for label, recs in recommendations.items():
        # Ensure the label is a valid string
        if isinstance(label, str):
            print(f"\n{label.capitalize()} Recommendations:")
        else:
            print("\nUnknown Label Recommendations:")
        for rec in recs:
            print(f"Brand: {rec['brand']}, Name: {rec['name']}, Price: {rec['price']}, URL: {rec['url']}")

