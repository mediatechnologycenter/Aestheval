import os
import json
import glob
from tqdm import tqdm
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from aestheval.text_prediction.predictor import Predictor
import configargparse
import os
import torch

tqdm.pandas()
dirname = os.path.dirname(__file__)

def config_parser():
    parser = configargparse.ArgumentParser(description="Training script")
    parser.add_argument("--reddit_dir", type=str, help="Data directory", default='data/RPCD')
    parser.add_argument("--only_predictions", action='store_true',default=False, help='Dataframe is already ready, run only predictions.')
    parser.add_argument("--no_attribute_prediction", action='store_false',default=True, help='Attribute prediction')
    parser.add_argument("--no_sentiment_prediction", action='store_false', default=True,help='sentiment_prediction')
    parser.add_argument("--no_emotion_prediction", action='store_false', default=True,help='Emotion prediction')

    return parser

# Method to read comment file given a submission by id
def read_comments_file(submission_id: str, subreddit:str, filename:str):
    subredditdirpath = os.path.join(root_dir, subreddit, 'comments', filename)
    submission_comments_csv_path = f"{submission_id}-comments.csv"
    submission_comments_path = os.path.join(subredditdirpath,submission_comments_csv_path)
    return pd.read_csv(submission_comments_path, index_col=None, header=0)

def load_reddit_dataset(subreddit_submissions):
    # Read every submission file a create dataframe with all submissions
    # Read comments from every submission and create dictionary comments[submission_id] = comments_dataframe
    print("Loading CSVs...")
    li = []
    comments = {}
    for submissions in subreddit_submissions:
        subreddit, filename = submissions.split("/")[-2:]
        filename = filename.split(".")[0]
        print(subreddit, filename)
        df = pd.read_csv(submissions, index_col=None, header=0)
        
        df['subreddit'] = subreddit
        
        li.append(df)

        for idx, submission in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                submission_comments = read_comments_file(submission_id=submission.id, subreddit=subreddit, filename=filename)
                comments[submission.id] = submission_comments
            except Exception as e:
                print(f"Not able to read comments from submission {submission.id} from {subreddit}")
                print(e)
                continue

    df = pd.concat(li, axis=0, ignore_index=True)

    return df, comments, subreddit

def filter_bad_comments(ids, texts):
    
    strings_to_filter = [
        "Your post has been marked as abandoned and has been removed.",
        "[deleted]",
        "I am a bot, and this action was performed automatically",
        "[removed]"
    ]
    
    filtered_ids, filtered_texts = [], []
    for id, text in zip(ids, texts):
        if isinstance(text, str) and not any([string in text for string in strings_to_filter]):
            filtered_ids.append(id)
            filtered_texts.append(text)
    return filtered_ids, filtered_texts

# First level comments are critiques in the photocritique dataset
def get_first_level_comments(submission, author=False):
    if submission.id not in comments.keys():
        return [], []

    def compare(val_a, val_b, author=False):
        if author:
            return val_a == val_b
        else:
            return val_a != val_b
    
    if not comments[submission.id].empty:
        # first level comments have the prefix t3
        condition = comments[submission.id].apply(lambda x: x.comment_id if compare(str(x.comment_author),submission.author_id, author=author) and str(x.comment_parent_id).split('_')[0] == "t3" else None, axis=1).notnull()
        return filter_bad_comments(comments[submission.id][condition].comment_id.to_list(), comments[submission.id][condition].comment_body.to_list())
    return [], []
  
def generate_comments_csv(comments, dataframe_path="raw_reddit_photocritique_comments.csv"):
    print("Generating comments file")
    li = []
    for submission_id in tqdm(comments):

        aux = comments[submission_id]
        aux['submission_id'] = submission_id
        aux['subreddit'] = 'photocritique'

        li.append(aux)

    print("Concatenating comments dfs...")
    comments_df = pd.concat(li, axis=0, ignore_index=True)  # Memory constrained
    comments_df.to_csv(dataframe_path, index=False)

def image_exists(images_paths, id):
    
    image_path = [image for image in images_paths if id in image]
    return len(image_path) != 0

def get_image_path(images_paths, id):
    image_paths = [image for image in images_paths if id in image]
    res = [path for path in image_paths if os.path.exists(path)]

    if res:
        return res[0].split('images/')[1]
    return None

def filter_bad_images(root_data, subreddit,  fn, edge_lenght_required=256):
    """Filter images corrupted or with longest edge shorter than a required length.
    """

    def has_file_allowed_extension(filename: str, extensions: "tuple[str]" =('jpeg', 'png', 'jpg')) -> bool:
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions)

    def is_valid_file(x: str) -> bool:
        return has_file_allowed_extension(x)
    if fn:
        path = os.path.join(root_data, subreddit, 'images', fn)
        # print(path)
        if not is_valid_file(path):
            return False
        try:
            image = Image.open(path)
            w, h= image.size
            image = image.resize((w,h))
            longest_edge = h if h>w else w
            return longest_edge > edge_lenght_required
        except Exception as e:
            print(e)
            return False
    return False

def prepare_dataframe(df, subreddit, dataframe_path):
    print(f"Raw number of posts: {len(df)}")
    print(f"Raw number of comments: {df.num_comments.sum()}")

    # Add created_utc columns with the correct data type
    df.created_utc = pd.to_datetime(df.created_utc, unit='s')
    print("Period of posts: ", df.created_utc.sort_values().min(),"->", df.created_utc.sort_values().max())

    # Get first level comments (critiques) and author (PO) comments
    df[["first_level_comments_ids", "first_level_comments_values"]] = pd.DataFrame(df.progress_apply(get_first_level_comments, axis=1).to_list(), index=df.index)
    df[["PO_comment_id","PO_comment_values"]] = pd.DataFrame(df.progress_apply(get_first_level_comments, author=True, axis=1).to_list(), index=df.index)

    print("Num of first level comments: ",sum(df["first_level_comments_ids"].apply(len)))
    print("Num of first level author comments: ",sum(df["PO_comment_id"].apply(len)))

    #Drop columns with constant values or missing all values
    df = df.drop(columns=['downs', 'awarders', 'top_awarded_type', 'approved_at_utc', 'num_reports', 'removed_by', 'view_count', 'selftext', 'ups']) 
    
    # Whether the post is from a discussion thread or not
    df["discussion_thread"] = df.title.apply(lambda x: "discussion thread" in x.lower())

    # Whether the posts have some kind of award or not
    df['awarded'] = df.gilded.apply(bool) | df.total_awards_received.apply(bool)

    # Whether the post is a "top post". Defined as a post which has been either awarded
    #  or has an score greater than the .75 percentile of posts with both comments and correct images
    df['top_post'] = df.awarded | (df.score > 18)

    # Create nsfw column
    df["nsfw"] = df.whitelist_status != "all_ads"
    df.drop(columns=["whitelist_status"], inplace=True)

    # Filter posts without image
    images = glob.glob(os.path.join(root_dir,"photocritique/images/*/**"), recursive=True)
    exists = df.apply(lambda x: image_exists(images_paths=images, id=x.id), axis=1)
    df["image_exists"] = exists
    print("Num of posts with images: ", len(df[df["image_exists"]]))

    # Filter posts without critiques
    comments_exist = df.first_level_comments_values.apply(len) > 0
    df["comments_exist"] = comments_exist
    print("Num of posts with comments: ", len(df[df["comments_exist"]]))
    
    # Get img paths
    im_paths = df.apply(lambda x: get_image_path(images_paths=images, id=x.id), axis=1)
    df['im_paths'] = im_paths

    # Filter posts with bad images
    print("Filtering bad images...")
    good = df.progress_apply(lambda x: filter_bad_images(root_data=root_dir, subreddit=subreddit, fn=x.im_paths), axis=1)
    df['image_good'] = good

    print(f"Num of posts with good images: {len(df[df['image_good']])}")

    df.to_pickle(dataframe_path)

    return df

def predict(model_name: str, df: pd.DataFrame) -> list:
    print("Predicting using model: ", model_name)
    results = []
    predictor = Predictor(model_path=model_name)
    for comments in tqdm(df['first_level_comments_values'].tolist()):
        results.append(predictor.predict(comments))
    return results

def process_predictions(list_of_predictions):
    return [{label['label']: label['score'] for label in elem} for elem in list_of_predictions]

if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    # Set data root directory and get all submissions files.
    root_dir = args.reddit_dir
    
    

    # Select which subreddits we want to analyze. We only have images of the photocritique subreddit for the moment.
    subreddits = ["photocritique"]

    submissions_dataframes = [glob.glob(f'{os.path.join(root_dir, subreddit)}/*.csv', recursive=True) for subreddit in subreddits]   
    # subreddits = ['photocritique', 'portraits', 'shittyHDR', 'postprocessing', 'photographs', 'AskPhotography']
    subreddit_submissions = [submissions_file for subreddit in submissions_dataframes for submissions_file in subreddit] 
    print(subreddit_submissions)
    

    print("Go and grab a coffe, this is going to take a while :)")
    if not args.only_predictions:
        df, comments, subreddit = load_reddit_dataset(subreddit_submissions)   
        df = prepare_dataframe(df, subreddit=subreddit, dataframe_path=os.path.join(root_dir,"reddit_photocritique_posts.pkl"))
        generate_comments_csv(comments, dataframe_path=os.path.join(root_dir,"raw_reddit_photocritique_comments.csv"))
    else:
        print("Reading existing dataframe...")
        df = pd.read_pickle(os.path.join(root_dir,"reddit_photocritique_posts.pkl"))
    
    if args.no_attribute_prediction:
        results = predict(model_name="daveni/aesthetic_attribute_classifier", df=df)
        df["aspect_prediction"] = results
        df['max_predicted_aspect'] = df.aspect_prediction.apply(lambda x: [max(elem, key= lambda key: elem[key]) for elem in x])
        df[["id", "aspect_prediction"]].to_pickle(os.path.join(root_dir,"aspect_predictions.pkl"))

    if args.no_sentiment_prediction:
        results = predict(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", df=df)
        df["sentiment_1"] = results
        df['max_sentiment_1'] = df.sentiment_1.apply(lambda x: [max(elem, key= lambda key: elem[key]) for elem in x])

        results = predict(model_name="siebert/sentiment-roberta-large-english", df=df)
        df["sentiment_2"] = results
        df['max_sentiment_2'] = df.sentiment_2.apply(lambda x: [max(elem, key= lambda key: elem[key]) for elem in x])

    if args.no_emotion_prediction:
        results = predict(model_name="j-hartmann/emotion-english-distilroberta-base", df=df)
        df["emotion"] = results
        df['max_emotion'] = df.emotion.apply(lambda x: [max(elem, key= lambda key: elem[key]) for elem in x])

    print(df.shape)


    # Add person predictions
    person_ids = pd.read_csv(os.path.join(dirname,'person_ids.csv'), header=None, names=['ids'])
    person_ids = person_ids.ids.apply(lambda x: x.split('_')[1].split('-')[0])
    pids = list(set(df.id).intersection(set(person_ids)))
    df["person_in_image"] = df.id.isin(pids)

    print(df.shape)


    # Add shot predictions
    with open(os.path.join(dirname, 'shot_pred.json'), 'r') as f:
        shot_data = json.load(f)
    shot_data = {k.split('submission_')[1].split('-')[0]: v for k,v in shot_data.items()}
    shot_df = pd.DataFrame.from_dict(shot_data, orient='index').reset_index()
    shot_df['max_shot_pred'] = shot_df[['ECS','CS', 'MS', 'FS', 'LS']].idxmax(axis=1)
    df = df.merge(shot_df, left_on='id', right_on='index').drop(columns=['index'])

    print(df.shape)

    # Add composition predictions
    # comp = torch.load(os.path.join(dirname,'output_composition.pth'))
    # composition = pd.DataFrame(comp['prob'].numpy(), columns=comp['classes'])
    # composition['max_predicted_composition'] = composition.idxmax(axis=1)
    # fixed_im_paths = [path.split('_')[1] for path in comp['fn']]
    # print(df.im_paths)
    # composition = pd.concat([composition, pd.DataFrame({'im_paths': fixed_im_paths})], axis=1)
    # df = df.merge(composition, on='im_paths')

    # print(df.shape)

    # # Add semantic prediction
    # sem = torch.load(os.path.join(dirname,'output_semantics.pth'))
    # semantics = pd.DataFrame(sem['prob'].numpy(), columns=sem['classes'])
    # semantics['max_predicted_semantic'] = semantics.idxmax(axis=1)
    # fixed_im_paths = [path.split('_')[1] for path in sem['fn']]
    # semantics = pd.concat([semantics, pd.DataFrame({'im_paths': fixed_im_paths})], axis=1)
    # df = df.merge(semantics, on='im_paths')

    print(df.shape)

    # Get only posts with both images and comments
    image_comments = df[df["image_exists"] & df["comments_exist"] & df["image_good"] & ~df["discussion_thread"]]
    image_comments = image_comments.drop(columns=['image_exists', 'comments_exist', 'image_good', 'discussion_thread']) 
    print("Num of posts with images and comments: ", len(image_comments))

    df.to_pickle(os.path.join(root_dir,"processed_reddit_dataset.pkl"))
    image_comments.to_pickle(os.path.join(root_dir,"image_and_comments_reddit_dataset.pkl"))

    image_comments = image_comments[['im_paths', 'aspect_prediction', 'first_level_comments_values']].to_dict('list')

    # Generate comments-images pairs json file
    with open(os.path.join(root_dir,'reddit_photocritique_image_comments.json'), 'w', encoding='utf-8') as f:
        json.dump(image_comments, f, ensure_ascii=False, indent=1)

    

    print("Done!")
