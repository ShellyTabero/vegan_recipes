import pandas as pd
from sentence_transformers import SentenceTransformer
import string
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from tqdm import tqdm
import pinecone
import cohere
import warnings
warnings.filterwarnings("ignore")
import re
import json

def load_and_embedd_dataset(
        file,
        text_fields,
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
):
    """
    Load a dataset and embed the text fields using a sentence-transformer model
    Args:
        file: The name of the file containing the data
        model: The model to use for embedding
        text_fields: The fields in the dataset that contain the text
    Returns:
        tuple: A tuple containing the dataset and the embeddings
    """
    dataset = pd.read_csv(file)
    translator = str.maketrans('', '', string.punctuation)

    # Apply the cleaning operation on the specified text columns to remove punctuation
    for field in text_fields:
        dataset[field] = dataset[field].apply(
            lambda x: x.translate(translator).replace('\n', ' ').lower() if isinstance(x, str) else x
        )

    # Concatenate the text fields into a single string per row
    dataset['combined_text'] = dataset[text_fields].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Embed the combined text
    embeddings = model.encode(dataset['combined_text'].tolist())
    return dataset, embeddings


def create_pinecone_index(
        index_name: str,
        dimension: int,
        metric: str = 'cosine',
):
    """
    Create a pinecone index if it does not exist
    Args:
        index_name: The name of the index
        dimension: The dimension of the index
        metric: The metric to use for the index
    Returns:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """
    print("Creating a Pinecone index...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc


def upsert_vectors(
        index,
        embeddings,
        dataset,
        text_fields,
        batch_size: int = 128
):
    """
    Upsert vectors to a pinecone index with multiple text fields as metadata.

    Args:
        index: The pinecone index object
        embeddings: The embeddings to upsert
        dataset: The dataset containing the metadata
        text_fields: The list of text fields in the dataset to include in metadata
        batch_size: The batch size to use for upserting

    Returns:
        An updated pinecone index
    """
    print("Upserting the embeddings to the Pinecone index...")
    shape = embeddings.shape
    ids = [str(i) for i in range(shape[0])]

    # Create a metadata entry combining the text fields for each vector
    meta = [
        {field: dataset[field][i] for field in text_fields}
        for i in range(shape[0])
    ]

    # Create a list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeddings, meta))

    # Upsert the vectors in batches
    for i in tqdm(range(0, shape[0], batch_size)):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])

    return index


def parse_text_to_json(recipe_text):
    # Extract index
    index_match = re.search(r'index:\s*(\d+)', recipe_text)
    index = int(index_match.group(1)) if index_match else None

    # Extract ingredients (assuming ingredients are provided as lists)
    ingredients_match = re.search(r'ingredients:\s*\[(.*?)\]', recipe_text)
    ingredients = [item.strip().strip('"') for item in ingredients_match.group(1).split(",")] if ingredients_match else []

    # Extract actions
    actions_match = re.search(r'actions:\s*\[(.*?)\]', recipe_text)
    actions = [item.strip().strip('"') for item in actions_match.group(1).split(",")] if actions_match else []

    # Extract tools
    tools_match = re.search(r'tools_required:\s*\[(.*?)\]', recipe_text)
    tools = [item.strip().strip('"') for item in tools_match.group(1).split(",")] if tools_match else []

    # Extract approximate time
    time_match = re.search(r'approximate_time:\s*(\d+)', recipe_text)
    approximate_time = int(time_match.group(1)) if time_match else None

    # Return as a structured dictionary (JSON-like)
    return {
        "index": index,
        "ingredients": ingredients,
        "actions": actions,
        "tools_required": tools,
        "approximate_time": approximate_time
    }


def add_json_to_df(json_row, df):
    # Initialize a dictionary to hold the new row's data
    row_data = {}

    # Add ingredient columns dynamically
    for ingredient in json_row["ingredients"]:
        if ingredient not in df.columns:
            df[ingredient] = 0  # Initialize new columns as 0
        row_data[ingredient] = 1  # Set to 1 for the current row

    # Add action columns dynamically
    for action in json_row["actions"]:
        if action not in df.columns:
            df[action] = 0  # Initialize new columns as 0
        row_data[action] = 1  # Set to 1 for the current row

    # Add tool columns dynamically
    for tool in json_row["tools_required"]:
        if tool not in df.columns:
            df[tool] = 0  # Initialize new columns as 0
        row_data[tool] = 1  # Set to 1 for the current row

    # Handle preparation time by putting it into a range
    if json_row["approximate_time"] is not None and json_row["approximate_time"] != 'null':
        time = int(json_row["approximate_time"])
        time_range = f"{(time // 10) * 10}-{(time // 10) * 10 + 10}"
        if time_range not in df.columns:
            df[time_range] = 0  # Initialize new columns as 0
        row_data[time_range] = 1  # Set to 1 for the current row

    # Add the row data to the DataFrame as a new row
    new_row_df = pd.DataFrame([row_data])  # Convert row_data to a DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)  # Use pd.concat() to append the new row
    print()
    return df



if __name__ == '__main__':
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    INDEX_NAME = 'finalproject'
    PINECONE_API_KEY = '5f0f9fd7-679b-4e02-b6b6-20ee52d3624f'
    COHERE_API_KEY = 'NCsTcmNpyxr7S1roTzelUO1QYB4RSlMFTgGZlZJK'
    co = cohere.Client(api_key=COHERE_API_KEY)

    # model = SentenceTransformer(EMBEDDING_MODEL)
    #
    # dataset, embeddings = load_and_embedd_dataset(
    #     file='data/vegan_recipes.csv',
    #     text_fields=['title', 'ingredients', 'preparation'],
    #     model=model
    # )
    # shape = embeddings.shape
    # pc = create_pinecone_index(INDEX_NAME, shape[1])
    # index = pc.Index(INDEX_NAME)
    # index_upserted = upsert_vectors(index, embeddings, dataset, ['title', 'ingredients', 'preparation'])
    #

    # df = pd.read_csv('data/vegan_recipes.csv').head(10)
    # new_df = pd.DataFrame()
    #
    # for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    #     query = f"""
    #                 I am going to give you a recipe, and I would like you to parse it in the following manner:
    #                 index: the number of the row
    #                 ingredients: [ingredient1, ingredient2, ...]
    #                 actions (for example baking, stirring...): [action1, action2...]
    #                 tools_required (for example oven, peeler..): [tool1, tool2...]
    #                 approximate time: total_time in minutes with no additional text. If you can't detect time in minutes,
    #                 put the string null
    #
    #                 Make sure the values do not repeat themselves.
    #                 Make sure that you write each ingredient in its singular and simplest form.
    #                 For instance, chopped tomatoes would be parsed as both chopped tomatoes and a tomato because we can
    #                 just chop the tomatoes, and so on. So in this case we would have in the ingredients both chopped tomatoes and tomatoes.
    #                 Also all the values should be lower-cased.
    #
    #                 For example:
    #                 recipe:
    #                 name: Simple Roasted Radish by ChicP
    #                 ingredients:
    #
    #                 1 170g tub beetroot and horseradish humus
    #                 300g mixed radishes
    #                 2 tbsp. light oil
    #                 1 tsp salt
    #
    #                 instructions:
    #
    #                 Preparation
    #                 Pre heat the oven to 160ֲ°C
    #                 Cut the radishes in half and place in a baking dish, drizzle with the oil and roast for about 20 minutes.
    #                 Spread the humus onto a serving plate.
    #                 Remove from the oven and season with salt.
    #                 Spoon on top of the beetroot and horseradish humus
    #
    #                 for this example the result would be:
    #                 {{
    #                     "index": 0
    #                     "ingredients": ["beetroot and horseradish houmous", "mixed radishes", "light oil", "salt"],
    #                     "actions": ["preheat", "cut", "place", "drizzle", "roast", "spread", "remove", "season", "spoon"],
    #                     "tools_required": ["oven", "baking dish", "serving plate", "spoon"],
    #                     "approximate_time": "20"
    #                 }}
    #
    #                 My recipe is:
    #                 Title: {row['title']}
    #                 Ingredients: {row['ingredients']}
    #                 Preparation: {row['preparation']}
    #                 For the answer please give us only the json, with no text before or after.
    #     """
    #     new_df = pd.read_csv('parsed_recipes.csv')
    #     response = co.chat(model='command-r-plus', message=query)
    #     #print('Query: ', query)
    #     #print(response.text)
    #     try:
    #         json_text = json.loads(response.text)
    #         new_df = add_json_to_df(json_text, new_df)
    #     except json.JSONDecodeError as e:
    #         # Handle JSON parsing errors
    #         print(f"JSONDecodeError: Failed to parse JSON - {e}")
    #         continue  # Skip to the next iteration
    #
    # print(i)
    # new_df.to_csv('parsed_recipes.csv')

    user_input = "I want to eat a lunch that doesn't take too long to make, that has vegetables like tomato"

    user_query = f"""
        I have the following user input: {user_input}
    
        Please infer the user's requirements and convert the input into the following structure:
    
        ingredients: Extract and list ingredients that the user explicitly mentions or implies, and infer which additional ingredients can be relevant..
        actions: Extract and list any cooking actions implied by the user's input (e.g., "chop", "fry"). If no actions are mentioned, leave the list empty.
        "tools_required": Extract and list any cooking actions implied
        time_requirements: Extract and convert any time-related phrases (e.g., "doesn't take too long", "quick", "long") into an approximate time in minutes.
        For the answer please give us only the json, with no text before or after.
        Make sure the values do not repeat themselves.
        Make sure that you write each ingredient in its singular and simplest form. 
        For instance, chopped tomatoes would be parsed as both chopped tomatoes and a tomato because we can 
        just chop the tomatoes, and so on. So in this case we would have in the ingredients both chopped tomatoes and tomatoes.
        Also all the values should be lower-cased.
    """
    new_df = pd.read_csv('parsed_recipes.csv')
    response_user = co.chat(model='command-r-plus', message=user_query)
    json_text = json.loads(response_user.text)
    new_df = add_json_to_df(json_text, new_df)
    new_df.to_csv('edited.csv')
    # for query in queries:
    #     co = cohere.Client(api_key=COHERE_API_KEY)
    #     response = co.chat(
    #         model='command-r-plus',
    #         message=query,
    #     )
    #     print('Query: ', query)
    #     print('\nAnswer: ', response.text)
    #     print('-------------------')

    # text = """{
    # "index": 0,
    # "ingredients": ["carrot ribbons", "frozen peas", "red cabbage", "yellow peppers", "red onion", "miso paste", "Chinese five spice", "soy sauce", "oil", "white rice"],
    # "actions": ["cook", "prep", "dice", "fry", "stir", "throw", "add"],
    # "tools_required": ["frying pan", "wok"],
    # "approximate_time": "10"
    # }
    # """
    # json_text = parse_text_to_json2(text)
    # print()
