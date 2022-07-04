# Based on:
# https://towardsdatascience.com/how-to-collect-a-reddit-dataset-c369de539114
# https://github.com/mattpodolak/pmaw
# https://praw.readthedocs.io/en/stable/getting_started/quick_start.html
# https://pushshift.io/api-parameters/
# Sample: https://api.pushshift.io/reddit/search/submission/?subreddit=portraits&size=10


import praw
import os
from dotenv import load_dotenv
import pandas as pd
from pmaw import PushshiftAPI
import datetime as dt
import time
from datetime import timedelta
import requests
from tqdm import tqdm
import glob
import csv
import ast

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT, ratelimit_seconds=600)
api = PushshiftAPI(praw=reddit)

subreddits = ['photocritique']


def log_action(action):
    print(action)
    return




def scrape_posts(data_dir: str):

    start_year = 2009
    end_year = 2022
    # start_year = args.year
    # end_year = start_year
    # directory on which to store the data
    basecorpus = data_dir

    LOG_EVERY = 100
    LIMIT = None



    ### BLOCK 1 ###

    for year in range(start_year, end_year+1):
        action = "[Year] " + str(year)
        log_action(action)

        dirpath = basecorpus + str(year)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # timestamps that define window of posts
        ts_after = int(dt.datetime(year, 1, 1).timestamp())
        ts_before = int(dt.datetime(year+1, 1, 1).timestamp())

    ### BLOCK 2 ###

        for subreddit in subreddits:
            start_time = time.time()

            action = "\t[Subreddit] " + subreddit
            log_action(action)

            subredditdirpath = dirpath + '/' + subreddit
            if os.path.exists(subredditdirpath):
                pass
            else:
                os.makedirs(subredditdirpath)

            submissions_csv_path = str(year) + '-' + subreddit + '-submissions.csv'
            
    ### BLOCK 3 ###

            submissions_dict = {
                "id" : [],
                "url" : [],
                "title" : [],
                "score" : [],
                "num_comments": [],
                "created_utc" : [],
                "selftext" : [],
                "author_id": [],
                "upvote_ratio": [],
                "ups": [],
                "downs": [],
                "gilded":[],
                "top_awarded_type": [],
                "total_awards_received": [],
                "all_awardings": [],
                "awarders": [],
                "approved_at_utc": [],
                "num_reports": [],
                "removed_by": [],
                "view_count": [],
                "preview": [],
                "num_crossposts": [],
                "link_flair_text": [],
                "whitelist_status":[]

            }

    ### BLOCK 4 ###

            # use PSAW only to get id of submissions in time interval
            gen = api.search_submissions(
                after=ts_after,
                before=ts_before,
                # filter=['id'],
                subreddit=subreddit,
                limit=LIMIT
            )

            print(api.metadata_)

    ### BLOCK 5 ###

            # use PMAW to get actual info and traverse comment tree
            for idx, submission_pmaw in enumerate(gen):

                submission_id = submission_pmaw['id']

                submissions_dict["id"].append(submission_pmaw['id'])
                submissions_dict["url"].append(submission_pmaw['url'])
                submissions_dict["title"].append(submission_pmaw['title'])
                submissions_dict["score"].append(submission_pmaw['score'])
                submissions_dict["num_comments"].append(submission_pmaw['num_comments'])
                submissions_dict["created_utc"].append(submission_pmaw['created_utc'])
                submissions_dict["selftext"].append(submission_pmaw['selftext'])
                submissions_dict["upvote_ratio"].append(submission_pmaw['upvote_ratio'])
                submissions_dict["ups"].append(submission_pmaw['ups'])
                submissions_dict["downs"].append(submission_pmaw['downs'])
                submissions_dict["gilded"].append(submission_pmaw['gilded'])
                submissions_dict["top_awarded_type"].append(submission_pmaw['top_awarded_type'])
                submissions_dict["total_awards_received"].append(submission_pmaw['total_awards_received'])
                submissions_dict["all_awardings"].append(submission_pmaw['all_awardings'])
                submissions_dict["awarders"].append(submission_pmaw['awarders'])
                submissions_dict["approved_at_utc"].append(submission_pmaw['approved_at_utc'])
                submissions_dict["num_reports"].append(submission_pmaw['num_reports'])
                submissions_dict["removed_by"].append(submission_pmaw['removed_by'])
                submissions_dict["view_count"].append(submission_pmaw['view_count'])
                submissions_dict["num_crossposts"].append(submission_pmaw['num_crossposts'])
                submissions_dict["link_flair_text"].append(submission_pmaw['link_flair_text'])
                submissions_dict["whitelist_status"].append(submission_pmaw['whitelist_status'])
                
                
                try:
                    submissions_dict["preview"].append(submission_pmaw['preview'])
                except:
                    submissions_dict["preview"].append(-1)
                try:
                    submissions_dict["author_id"].append(submission_pmaw['author'].id)
                except:
                    submissions_dict["author_id"].append(-1)

                if idx % LOG_EVERY == 0 and idx>0:
                    time_delta = time.time() - start_time
                    action = f"\t\t[Info] {idx} submissions processed after {timedelta(seconds=time_delta)}"
                    log_action(action)

    ### BLOCK 6 ###

            # single csv file with all submissions
            pd.DataFrame(submissions_dict).to_csv(subredditdirpath + '/' + submissions_csv_path,
                                                index=False)


            action = f"\t\t[Info] Found submissions: {pd.DataFrame(submissions_dict).shape[0]}"
            log_action(action)

            action = f"\t\t[Info] Elapsed time: {time.time() - start_time: .2f}s"
            log_action(action)

def scrape_comments(data_dir: str):

    basecorpus = data_dir


    submissions_dataframes = glob.glob(f'{basecorpus}/*/*/*.csv', recursive=True)
    print(submissions_dataframes)

    subreddit_submissions = [submissions for submissions in submissions_dataframes for subreddit in subreddits if subreddit in submissions]   

    

    for submissions in sorted(subreddit_submissions):
        print(submissions)
        year, subreddit = submissions.split(basecorpus)[1].split('/')[0:2]
        

        action = "[Year] " + str(year) + " [Subreddit] " + subreddit
        log_action(action)
        submissions_df = pd.read_csv(submissions)

        dirpath = os.path.join(basecorpus, str(year))
        subredditdirpath = os.path.join(dirpath,subreddit, 'comments')

        if not os.path.exists(subredditdirpath):
            os.makedirs(subredditdirpath)

        print(subredditdirpath)
        
        for idx, submission in tqdm(submissions_df.iterrows(), total=submissions_df.shape[0]):
            submission_id = submission.id
            submission_comments_csv_path = str(year) + '-' + subreddit + '-submission_' + submission_id + '-comments.csv'
            submission_comments_path = os.path.join(subredditdirpath, submission_comments_csv_path)
            # if os.path.exists(submission_comments_path):
            #     continue
            
            submission_comments_dict = {
                "comment_id" : [],
                "comment_parent_id" : [],
                "comment_body" : [],
                "comment_link_id" : [],
                "comment_score": [],
                "comment_all_awardings": [],
                "comment_top_awarded_type": [],
                "comment_total_awards_received": [],
                'comment_controversiality': [],
                'comment_depth': [],
                "comment_ups": [],
                "comment_downs": [],
                "comment_gilded":[],
                "comment_score": [],
                "comment_author": [],
                "created_utc": []
            }

        ### BLOCK 7 ###

            # extend the comment tree all the way
            submission_praw = reddit.submission(id=submission_id)
            submission_praw.comments.replace_more(limit=None)
            # comment_ids = api.search_submission_comment_ids(ids=submission_id)
            # comment_id_list = [c_id for c_id in comment_ids]
            # comments = api.search_comments(ids=comment_id_list)
            # comment_list = [comment for comment in comments]
            # for each comment in flattened comment tree
            for comment in submission_praw.comments.list():
                submission_comments_dict["comment_id"].append(comment.id)
                submission_comments_dict["comment_parent_id"].append(comment.parent_id)
                submission_comments_dict["comment_body"].append(comment.body)
                submission_comments_dict["comment_link_id"].append(comment.link_id)
                submission_comments_dict["comment_all_awardings"].append(comment.all_awardings)
                submission_comments_dict["comment_top_awarded_type"].append(comment.top_awarded_type)
                submission_comments_dict["comment_controversiality"].append(comment.controversiality)
                submission_comments_dict["comment_depth"].append(comment.depth)
                submission_comments_dict["comment_ups"].append(comment.ups)
                submission_comments_dict["comment_downs"].append(comment.downs)
                submission_comments_dict["comment_gilded"].append(comment.gilded)
                submission_comments_dict["comment_score"].append(comment.score)
                submission_comments_dict["comment_total_awards_received"].append(comment.total_awards_received)
                submission_comments_dict["created_utc"].append(comment.created_utc)

                try:
                    submission_comments_dict["comment_author"].append(comment.author.id)
                except:
                    submission_comments_dict["comment_author"].append(-1)


            # for each submission save separate csv comment file
            pd.DataFrame(submission_comments_dict).to_csv(submission_comments_path, index=False)


def log_bad_image(data_dir, submission_id: str) -> None:
        
    logging_dir = f"{data_dir}/logs"
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    
    with open(os.path.join(logging_dir, "images_not_ok.csv"), "a") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(submission_id)

def scrape_images(data_dir: str):

    basecorpus = data_dir
    processed_posts_file = os.path.join(basecorpus, "reddit_photocritique_posts.pkl")

    # If processed file already exist, use it to download images. Else, use scraped posts.
    if os.path.exists(processed_posts_file):
        print(f"Reddit posts from {processed_posts_file}")
        submissions_df = pd.read_pickle(processed_posts_file)
        download_images_from_df(data_dir, submissions_df)

    else:
        submissions_dataframes = glob.glob(f'{basecorpus}/*/*/*.csv', recursive=True)
        print("Getting images for:\n", submissions_dataframes)



        subreddit_submissions = [submissions for submissions in submissions_dataframes for subreddit in subreddits if subreddit in submissions]   


        for submissions in subreddit_submissions:

            year, subreddit = submissions.split(basecorpus)[1].split('/')[0:2]

            action = "[Year] " + str(year) + " [Subreddit] " + subreddit
            log_action(action)
            submissions_df = pd.read_csv(submissions)
            
            download_images_from_df(data_dir, submissions_df)

def download_images_from_df(data_dir: str, submissions_df:pd.DataFrame):


    for idx, submission in tqdm(submissions_df.iterrows(), total=submissions_df.shape[0]):
        submission_id = submission.id
        dirpath = data_dir + str(submission.year)


        subredditdirpath = os.path.join(dirpath, submission.subreddit, 'images')
        if not os.path.exists(subredditdirpath):
            os.makedirs(subredditdirpath)
        
        submission_image_path = f"{submission.year}-{submission.subreddit}-submission_" +submission_id +'-image'
        submission_images_path = os.path.join(subredditdirpath,submission_image_path)
        if glob.glob(submission_images_path + ".*"): #os.path.exists(submission_images_path + ".*"):
            # print("path exists, not downloading images again")
            continue
        
        preview = ast.literal_eval(submission.preview)
        
        try:
            response = requests.get(preview['images'][0]['source']['url'], stream=True)
        except Exception as first_e:
            # print("Couldn't get source image from preview")
            # print(e)
            try:
                response = requests.get(submission.url, stream=True)
            except Exception as e:
                print("Couldn't get source image from either preview or url")
                print("First E", first_e)
                print(e)
                log_bad_image(data_dir, submission_id)
                continue   
        

        if not response.ok:
            print(response)
            log_bad_image(data_dir, submission_id)
        try:
            _, extension = response.headers["content-type"].split("/")
        except Exception as e:
            print(e)
            log_bad_image(data_dir, submission_id)
            continue
        try:
            with open(f'{submission_images_path}.{extension}', 'wb') as handle:
                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)
        except Exception as e:
            print(e)
            log_bad_image(data_dir, submission_id)
            continue

